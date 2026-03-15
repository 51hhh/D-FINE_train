"""
D-FINE: Redefine Regression Task of DETRs as Fine-grained Distribution Refinement
Copyright (c) 2024 The D-FINE Authors. All Rights Reserved.
---------------------------------------------------------------------------------
Modified from RT-DETR (https://github.com/lyuwenyu/RT-DETR)
Copyright (c) 2023 lyuwenyu. All Rights Reserved.
"""

import copy
import datetime
import json
import time
import threading

import torch

from ..misc import dist_utils, stats
from ._solver import BaseSolver
from .det_engine import evaluate, evaluate_overactivation, train_one_epoch


class DetSolver(BaseSolver):
    def fit(self):
        self.train()
        args = self.cfg
        metric_names = ["AP50:95", "AP50", "AP75", "APsmall", "APmedium", "APlarge"]
        _oa_thread = None   # 后台 OA 评估线程
        _oa_result = {}     # 上一轮 OA 结果（在下一轮训练开始后写入 TB）
        _best_oa_max = 1.0  # 最优 oa_max_score（越低越好），用于 stg2 联合保存决策

        if self.use_wandb:
            import wandb

            wandb.init(
                project=args.yaml_cfg["project_name"],
                name=args.yaml_cfg["exp_name"],
                config=args.yaml_cfg,
            )
            wandb.watch(self.model)

        n_parameters, model_stats = stats(self.cfg)
        print(model_stats)
        print("-" * 42 + "Start training" + "-" * 43)
        top1 = 0
        best_stat = {
            "epoch": -1,
        }
        if self.last_epoch > 0:
            module = self.ema.module if self.ema else self.model
            test_stats, coco_evaluator = evaluate(
                module,
                self.criterion,
                self.postprocessor,
                self.val_dataloader,
                self.evaluator,
                self.device,
                self.last_epoch,
                self.use_wandb
            )
            for k in test_stats:
                best_stat["epoch"] = self.last_epoch
                best_stat[k] = test_stats[k][0]
                top1 = test_stats[k][0]
                print(f"best_stat: {best_stat}")

        best_stat_print = best_stat.copy()
        start_time = time.time()
        start_epoch = self.last_epoch + 1
        for epoch in range(start_epoch, args.epochs):
            self.train_dataloader.set_epoch(epoch)
            # self.train_dataloader.dataset.set_epoch(epoch)
            if dist_utils.is_dist_available_and_initialized():
                self.train_dataloader.sampler.set_epoch(epoch)

            if epoch == self.train_dataloader.collate_fn.stop_epoch:
                self.load_resume_state(str(self.output_dir / "best_stg1.pth"))
                if self.ema:
                    self.ema.decay = self.train_dataloader.collate_fn.ema_restart_decay
                    print(f"Refresh EMA at epoch {epoch} with decay {self.ema.decay}")

            train_stats = train_one_epoch(
                self.model,
                self.criterion,
                self.train_dataloader,
                self.optimizer,
                self.device,
                epoch,
                epochs=args.epochs,
                max_norm=args.clip_max_norm,
                print_freq=args.print_freq,
                ema=self.ema,
                scaler=self.scaler,
                lr_warmup_scheduler=self.lr_warmup_scheduler,
                writer=self.writer,
                use_wandb=self.use_wandb,
                output_dir=self.output_dir,
            )

            if self.lr_warmup_scheduler is None or self.lr_warmup_scheduler.finished():
                self.lr_scheduler.step()

            self.last_epoch += 1

            if self.output_dir and epoch < self.train_dataloader.collate_fn.stop_epoch:
                checkpoint_paths = [self.output_dir / "last.pth"]
                # extra checkpoint before LR drop and every 100 epochs
                if (epoch + 1) % args.checkpoint_freq == 0:
                    checkpoint_paths.append(self.output_dir / f"checkpoint{epoch:04}.pth")
                for checkpoint_path in checkpoint_paths:
                    dist_utils.save_on_master(self.state_dict(), checkpoint_path)

            module = self.ema.module if self.ema else self.model
            test_stats, coco_evaluator = evaluate(
                module,
                self.criterion,
                self.postprocessor,
                self.val_dataloader,
                self.evaluator,
                self.device,
                epoch,
                self.use_wandb,
                output_dir=self.output_dir,
            )

            _prev_oa_result = {}  # 保存本轮可用的OA结果（来自上一轮线程）
            if self.cfg.yaml_cfg.get("eval_overactivation", False) and dist_utils.is_main_process():
                neg_dir = self.cfg.yaml_cfg.get("negative_img_dir", "")
                oa_input_size = self.cfg.yaml_cfg.get("eval_spatial_size", [640, 640])
                oa_input_size = oa_input_size[0] if isinstance(oa_input_size, (list, tuple)) else oa_input_size
                oa_threshold = self.cfg.yaml_cfg.get("oa_conf_threshold", 0.3)
                # 等待上一轮 OA 线程完成，记录结果
                if _oa_thread is not None:
                    _oa_thread.join()
                    if _oa_result and self.writer:
                        for k, v in _oa_result.items():
                            self.writer.add_scalar(f"Test/{k}", v, epoch - 1)
                        print(f"Over-Activation @{oa_threshold} [epoch {epoch-1}]: {_oa_result}")
                    _prev_oa_result = dict(_oa_result)  # 保存供 rollback 逻辑使用
                    _oa_result.clear()
                # 启动本轮 OA 评估（后台线程，下一轮训练开始后才等待结果）
                _oa_model = copy.deepcopy(module).eval()
                _oa_postprocessor = copy.deepcopy(self.postprocessor)
                def _run_oa(_m=_oa_model, _pp=_oa_postprocessor, _nd=neg_dir,
                            _dev=self.device, _sz=oa_input_size, _thr=oa_threshold):
                    m = evaluate_overactivation(_m, _pp, _nd, _dev, input_size=_sz, conf_threshold=_thr)
                    _oa_result.update(m)
                _oa_thread = threading.Thread(target=_run_oa, daemon=True)
                _oa_thread.start()

            # TODO
            for k in test_stats:
                if self.writer and dist_utils.is_main_process():
                    for i, v in enumerate(test_stats[k]):
                        self.writer.add_scalar(f"Test/{k}_{i}".format(k), v, epoch)

                if k in best_stat:
                    best_stat["epoch"] = (
                        epoch if test_stats[k][0] > best_stat[k] else best_stat["epoch"]
                    )
                    best_stat[k] = max(best_stat[k], test_stats[k][0])
                else:
                    best_stat["epoch"] = epoch
                    best_stat[k] = test_stats[k][0]

                if best_stat[k] > top1:
                    best_stat_print["epoch"] = epoch
                    top1 = best_stat[k]
                    if self.output_dir:
                        if epoch >= self.train_dataloader.collate_fn.stop_epoch:
                            dist_utils.save_on_master(
                                self.state_dict(), self.output_dir / "best_stg2.pth"
                            )
                        else:
                            dist_utils.save_on_master(
                                self.state_dict(), self.output_dir / "best_stg1.pth"
                            )

                best_stat_print[k] = max(best_stat[k], top1)
                print(f"best_stat: {best_stat_print}")  # global best

                if best_stat["epoch"] == epoch and self.output_dir:
                    if epoch >= self.train_dataloader.collate_fn.stop_epoch:
                        if test_stats[k][0] > top1:
                            top1 = test_stats[k][0]
                            dist_utils.save_on_master(
                                self.state_dict(), self.output_dir / "best_stg2.pth"
                            )
                    else:
                        top1 = max(test_stats[k][0], top1)
                        dist_utils.save_on_master(
                            self.state_dict(), self.output_dir / "best_stg1.pth"
                        )

                elif epoch >= self.train_dataloader.collate_fn.stop_epoch:
                    # stg2 AP 未提升：检查 oa_max_score 是否有改善。
                    # 若 oa_max_score 下降超过阈值，保存为 best_oa.pth 并跳过 rollback。
                    # 否则按原逻辑 rollback 到 best_stg1.pth。
                    cur_oa_max = _prev_oa_result.get("oa_max_score", 1.0)
                    oa_ap_tolerance = self.cfg.yaml_cfg.get("oa_ap_tolerance", 0.0)
                    ap_close_enough = (top1 - test_stats[k][0]) <= oa_ap_tolerance
                    if cur_oa_max < _best_oa_max and ap_close_enough:
                        _best_oa_max = cur_oa_max
                        if self.output_dir and dist_utils.is_main_process():
                            dist_utils.save_on_master(
                                self.state_dict(), self.output_dir / "best_oa.pth"
                            )
                            print(f"[OA] Saved best_oa.pth at epoch {epoch}: "
                                  f"oa_max={cur_oa_max:.4f} (AP={test_stats[k][0]:.4f}, "
                                  f"top1={top1:.4f}, delta={top1-test_stats[k][0]:.4f})")
                    else:
                        best_stat = {
                            "epoch": -1,
                        }
                        if self.ema:
                            self.ema.decay -= 0.0001
                            self.load_resume_state(str(self.output_dir / "best_stg1.pth"))
                            print(f"Refresh EMA at epoch {epoch} with decay {self.ema.decay}")

            log_stats = {
                **{f"train_{k}": v for k, v in train_stats.items()},
                **{f"test_{k}": v for k, v in test_stats.items()},
                "epoch": epoch,
                "n_parameters": n_parameters,
            }

            if self.use_wandb:
                wandb_logs = {}
                for idx, metric_name in enumerate(metric_names):
                    wandb_logs[f"metrics/{metric_name}"] = test_stats["coco_eval_bbox"][idx]
                wandb_logs["epoch"] = epoch
                wandb.log(wandb_logs)

            if self.output_dir and dist_utils.is_main_process():
                with (self.output_dir / "log.txt").open("a") as f:
                    f.write(json.dumps(log_stats) + "\n")

                # for evaluation logs
                if coco_evaluator is not None:
                    (self.output_dir / "eval").mkdir(exist_ok=True)
                    if "bbox" in coco_evaluator.coco_eval:
                        filenames = ["latest.pth"]
                        if epoch % 50 == 0:
                            filenames.append(f"{epoch:03}.pth")
                        for name in filenames:
                            torch.save(
                                coco_evaluator.coco_eval["bbox"].eval,
                                self.output_dir / "eval" / name,
                            )

        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        print("Training time {}".format(total_time_str))
        # 等待最后一轮 OA 线程完成并写入 TB
        if _oa_thread is not None:
            _oa_thread.join()
            if _oa_result and self.writer and dist_utils.is_main_process():
                _final_thr = self.cfg.yaml_cfg.get("oa_conf_threshold", 0.3)
                for k, v in _oa_result.items():
                    self.writer.add_scalar(f"Test/{k}", v, args.epochs - 1)
                print(f"Over-Activation @{_final_thr} [final]: {_oa_result}")

    def val(self):
        self.eval()

        module = self.ema.module if self.ema else self.model
        test_stats, coco_evaluator = evaluate(
            module,
            self.criterion,
            self.postprocessor,
            self.val_dataloader,
            self.evaluator,
            self.device,
            epoch=-1,
            use_wandb=False,
        )

        if self.cfg.yaml_cfg.get("eval_overactivation", False) and dist_utils.is_main_process():
            neg_dir = self.cfg.yaml_cfg.get("negative_img_dir", "")
            oa_input_size = self.cfg.yaml_cfg.get("eval_spatial_size", [640, 640])
            oa_input_size = oa_input_size[0] if isinstance(oa_input_size, (list, tuple)) else oa_input_size
            oa_threshold = self.cfg.yaml_cfg.get("oa_conf_threshold", 0.3)
            oa_metrics = evaluate_overactivation(
                module, self.postprocessor, neg_dir, self.device,
                input_size=oa_input_size, conf_threshold=oa_threshold,
            )
            if oa_metrics:
                print(f"Over-Activation @{oa_threshold}: {oa_metrics}")

        if self.output_dir:
            dist_utils.save_on_master(
                coco_evaluator.coco_eval["bbox"].eval, self.output_dir / "eval.pth"
            )

        return
