import os
import traceback
from collections import OrderedDict

import torch

import data
from evaluator.evaluation import evaluate_training_set, \
    evaluate_validation_set, inference_validation
from managers.inference_manager import InferenceManager
from managers.trainer_manager import TrainerManager
from options.train_options import TrainOptions
from util.files import copy_src
from util.iter_counter import IterationCounter
from util.visualizer import Visualizer


def run(opt):
    print("Number of GPUs used: {}".format(torch.cuda.device_count()))
    print("Current Experiment Name: {}".format(opt.name))

    # The dataloader will yield the training samples
    dataloader = data.create_dataloader(opt)

    trainer = TrainerManager(opt)
    inference_manager = InferenceManager(num_samples=opt.num_evaluation_samples, opt=opt, cuda=len(opt.gpu_ids) > 0, write_details=False, save_images=False, use_metrics=False)

    # For logging and visualizations
    iter_counter = IterationCounter(opt, len(dataloader))
    visualizer = Visualizer(opt)

    if not opt.debug:
        # We keep a copy of the current source code for each experiment
        copy_src(path_from="./",
                 path_to=os.path.join(opt.checkpoints_dir, opt.name))

    # We wrap training into a try/except clause such that the model is saved
    # when interrupting with Ctrl+C
    try:
        for epoch in iter_counter.training_epochs():
            iter_counter.record_epoch_start(epoch)
            for i, data_i in enumerate(dataloader, start=iter_counter.epoch_iter):

                # Training the generator
                if i % opt.D_steps_per_G == 0:
                    trainer.run_generator_one_step(data_i)

                # Training the discriminator
                trainer.run_discriminator_one_step(data_i)

                iter_counter.record_one_iteration()

                # Logging, plotting and visualizing
                if iter_counter.needs_printing():
                    losses = trainer.get_latest_losses()
                    visualizer.print_current_errors(epoch, iter_counter.epoch_iter,
                                                    losses,
                                                    iter_counter.time_per_iter,
                                                    iter_counter.total_time_so_far,
                                                    iter_counter.total_steps_so_far)
                    visualizer.plot_current_errors(losses,
                                                   iter_counter.total_steps_so_far)

                if iter_counter.needs_displaying():
                    logs = trainer.get_logs()
                    visuals = [
                        ('input_label', data_i['label']),
                        ('out_train', trainer.get_latest_generated()),
                        ('real_train', data_i['image'])
                    ]
                    if opt.guiding_style_image:
                        visuals.append(('guiding_image', data_i['guiding_image']))
                        visuals.append(
                            ('guiding_input_label', data_i['guiding_label']))

                    if opt.evaluate_val_set:
                        validation_output = inference_validation(trainer.sr_model,
                                                                 inference_manager,
                                                                 opt)
                        visuals += validation_output
                    visuals = OrderedDict(visuals)
                    visualizer.display_current_results(visuals, epoch,
                                                       iter_counter.total_steps_so_far,
                                                       logs)

                if iter_counter.needs_saving():
                    print('Saving the latest model (epoch %d, total_steps %d)' %
                          (epoch, iter_counter.total_steps_so_far))
                    trainer.save('latest')
                    iter_counter.record_current_iter()

                if iter_counter.needs_evaluation():
                    # Evaluate on training set
                    result_train = evaluate_training_set(inference_manager,
                                                         trainer.sr_model_on_one_gpu,
                                                         dataloader)
                    info = iter_counter.record_fid(result_train["FID"],
                                                   split="train",
                                                   num_samples=opt.num_evaluation_samples)
                    info += os.linesep + iter_counter.record_metrics(result_train,
                                                                     split="train")
                    visualizer.plot_current_errors(result_train,
                                                   iter_counter.total_steps_so_far,
                                                   split="train/")

                    if opt.evaluate_val_set:
                        # Evaluate on validation set
                        result_val = evaluate_validation_set(inference_manager,
                                                             trainer.sr_model_on_one_gpu,
                                                             opt)
                        info += os.linesep + iter_counter.record_fid(
                            result_val["FID"], split="validation",
                            num_samples=opt.num_evaluation_samples)
                        info += os.linesep + iter_counter.record_metrics(
                            result_val, split="validation")
                        visualizer.plot_current_errors(result_val,
                                                       iter_counter.total_steps_so_far,
                                                       split="validation/")

            trainer.update_learning_rate(epoch)
            iter_counter.record_epoch_end()

            if epoch % opt.save_epoch_freq == 0 or \
                            epoch == iter_counter.total_epochs:
                print('Saving the model at the end of epoch %d, iters %d' %
                      (epoch, iter_counter.total_steps_so_far))
                trainer.save('latest')
                trainer.save(epoch)
                iter_counter.record_current_iter()

        print('Training was successfully finished.')
    except (KeyboardInterrupt, SystemExit):
        print("KeyboardInterrupt. Shutting down.")
        print(traceback.format_exc())
    except Exception as e:
        print(traceback.format_exc())
    finally:
        print('Saving the model before quitting')
        trainer.save('latest')
        iter_counter.record_current_iter()


if __name__ == "__main__":
    # Parse arguments
    opt = TrainOptions().parse()
    run(opt)
