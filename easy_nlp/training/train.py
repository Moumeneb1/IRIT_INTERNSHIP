import torch
import random
import numpy as np
from .metrics import flat_accuracy, flat_f1, flat_precision, flat_recall
import sys

import time
import datetime


class Train:
    def __format_time(elapsed):
        '''
        Takes a time in seconds and returns a string hh:mm:ss
        '''
        # Round to the nearest second.
        elapsed_rounded = int(round((elapsed)))

        # Format as hh:mm:ss
        return str(datetime.timedelta(seconds=elapsed_rounded))

    def fit(model, train_dataloader, validation_dataloader, epochs, device, optimizer, scheduler, criterion, writer=0, print_each=40):
        # Set the seed value all ovr the place to make this reproducible.
        seed_val = 2

        random.seed(seed_val)
        np.random.seed(seed_val)
        torch.manual_seed(seed_val)
        torch.cuda.manual_seed_all(seed_val)
        model_save_path = 'tmp'
        loss_values = []
        hist_valid_scores = []

        # For each epoch...
        for epoch_i in range(0, epochs):
            logs = {}

            # ========================================
            #               Training
            # ========================================

            # Perform one full pass over the training set.

            print("")
            print(
                '======== Epoch {:} / {:} ========'.format(epoch_i + 1, epochs))
            print('Training...')

            # Measure how long the training epoch takes.
            t0 = time.time()

            # Reset the total loss for this epoch.
            total_loss = 0
            total_accuracy = 0

            # Put the model into training mode. Don't be mislead--the call to
            # `train` just changes the *mode*, it doesn't *perform* the training.
            model.train()

            # For each batch of training data...
            for step, batch in enumerate(train_dataloader):

                # Progress update every 40 batches.
                if step % print_each == 0 and not step == 0:
                    # Calculate elapsed time in minutes.
                    elapsed = format_time(time.time() - t0)

                    # Report progress.
                    print('  Batch {:>5,}  of  {:>5,}.    Elapsed: {:}.'.format(
                        step, len(train_dataloader), elapsed))

                # move batch data to device (cpu or gpu)
                batch = tuple(t.to(device) for t in batch)

                # `batch` contains three pytorch tensors:
                #   [0]: input ids
                #   [1]: attention masks
                #   [2]: labels

                model.zero_grad()
                outputs = model(batch)

                # The call to `model` always returns a tuple, so we need to pull the
                # loss value out of the tuple.
                logits = outputs[0]
                label_ids = batch[-1]
                # print(logits)

                loss = criterion(logits.view(-1, model.n_class),
                                 label_ids.view(-1))

                # Move logits back to cpu for metrics calculations
                logits = logits.detach().cpu().numpy()
                label_ids = label_ids.to('cpu').numpy()

                # Calculate the accuracy for this batch of test sentences.
                current_accuracy = flat_accuracy(logits, label_ids)
                total_accuracy += current_accuracy
                # Accumulate the training loss over all of the batches so that we can
                # calculate the average loss at the end. `loss` is a Tensor containing a
                # single value; the `.item()` function just returns the Python value
                # from the tensor.
                total_loss += loss.item()
                writer.add_scalar('training loss',
                                  loss.item(),
                                  epoch_i * len(train_dataloader)+step)
                writer.add_scalar('training Accuracy',
                                  current_accuracy,
                                  epoch_i * len(train_dataloader)+step)

                # Perform a backward pass to calculate the gradients.
                loss.backward()

                # Clip the norm of the gradients to 1.0.
                # This is to help prevent the "exploding gradients" problem.
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

                # Update parameters and take a step using the computed gradient.
                # The optimizer dictates the "update rule"--how the parameters are
                # modified based on their gradients, the learning rate, etc.
                optimizer.step()

                # Update the learning rate.
                scheduler.step()

            # Calculate the average loss over the training data.
            avg_train_loss = total_loss / len(train_dataloader)
            avg_train_accuracy = total_accuracy / len(train_dataloader)
            # Store the loss value for plotting the learning curve.
            loss_values.append(avg_train_loss)
            logs["log loss"] = avg_train_loss
            logs["accuracy"] = avg_train_accuracy

            print("")
            print("  Average training loss: {0:.2f}".format(avg_train_loss))
            print("  Training epcoh took: {:}".format(
                format_time(time.time() - t0)))

            # ========================================
            #               Validation
            # ========================================
            # After the completion of each training epoch, measure our performance on
            # our validation set.

            print("")
            print("Running Validation...")

            t0 = time.time()

            # Put the model in evaluation mode--the dropout layers behave differently
            # during evaluation.
            model.eval()

            # Tracking variables
            eval_loss, eval_accuracy, eval_f1, eval_recall, eval_precesion = 0, 0, 0, 0, 0
            nb_eval_steps, nb_eval_examples = 0, 0
            validation_loss = 0
            # Evaluate data for one epoch
            for step_valid, batch in enumerate(validation_dataloader):

                # Add batch to GPU
                batch = tuple(t.to(device) for t in batch)

                # Unpack the inputs from our dataloader

                # Telling the model not to compute or store gradients, saving memory and
                # speeding up validation
                with torch.no_grad():

                    # Forward pass, calculate logit predictions.
                    # This will return the logits rather than the loss because we have
                    # not provided labels.
                    # token_type_ids is the same as the "segment ids", which
                    # differentiates sentence 1 and 2 in 2-sentence tasks.
                    # The documentation for this `model` function is here:
                    model.eval()
                    # https://huggingface.co/transformers/v2.2.0/model_doc/bert.html#transformers.BertForSequenceClassification
                    outputs = model(batch)

                # Get the "logits" output by the model. The "logits" are the output
                # values prior to applying an activation function like the softmax.

                logits = outputs[0]
                label_ids = batch[-1]

                validation_loss += criterion(logits.view(-1,
                                                         model.n_class), label_ids.view(-1))

                # print(logits)

                # Move logits and labels to CPU
                logits = logits.detach().cpu().numpy()
                label_ids = label_ids.to('cpu').numpy()

                # Calculate the accuracy for this batch of test sentences.
                tmp_eval_accuracy = flat_accuracy(logits, label_ids)
                tmp_eval_f1 = flat_f1(logits, label_ids)
                tmp_eval_recall = flat_recall(logits, label_ids)
                tmp_eval_precision = flat_precision(logits, label_ids)
                # Accumulate the total scores.
                eval_accuracy += tmp_eval_accuracy
                eval_f1 += tmp_eval_f1
                eval_recall += tmp_eval_recall
                eval_precesion += tmp_eval_precision

                # Track the number of batches
                nb_eval_steps += 1
                validation_loss = validation_loss/len(validation_dataloader)
                writer.add_scalar('validation Accuracy',
                                  tmp_eval_accuracy,
                                  epoch_i * len(validation_dataloader)+step_valid)
                writer.add_scalar('validation F1',
                                  tmp_eval_f1,
                                  epoch_i * len(validation_dataloader)+step_valid)
                writer.add_scalar('validation recall',
                                  tmp_eval_recall,
                                  epoch_i * len(validation_dataloader)+step_valid)
                writer.add_scalar('validation precesion',
                                  tmp_eval_precision,
                                  epoch_i * len(validation_dataloader)+step_valid)
            is_better = len(hist_valid_scores) == 0 or validation_loss < min(
                hist_valid_scores)
            hist_valid_scores.append(validation_loss)

            if is_better:
                patience = 0
                print(
                    'save currently the best model to [%s]' % model_save_path, file=sys.stderr)
                model.save(model_save_path)
                # also save the optimizers' state
                torch.save(optimizer.state_dict(), model_save_path + '.optim')
            elif patience < 5:
                patience += 1
                '''print('hit patience %d' % patience, file=sys.stderr)
            if patience == int(5):
                # decay lr, and restore from previously best checkpoint
                print('load previously best model and decay learning rate to ', file=sys.stderr)
                # load model
                params = torch.load(model_save_path, map_location=lambda storage, loc: storage)
                model.load_state_dict(params['state_dict'])
                model = model.to(torch.device("cuda"))
                print('restore parameters of the optimizers', file=sys.stderr)
                optimizer.load_state_dict(torch.load(model_save_path + '.optim'))
                # reset patience
                patience = 0'''
            # Report the final accuracy for this validation run.
            print("  Accuracy: {0:.2f}".format(eval_accuracy/nb_eval_steps))
            print("  F1: {0:.2f}".format(eval_f1/nb_eval_steps))
            print("  Recall: {0:.2f}".format(eval_recall/nb_eval_steps))
            print("  Precision: {0:.2f}".format(eval_precesion/nb_eval_steps))
            print("  Validation took: {:}".format(
                format_time(time.time() - t0)))
        return (eval_accuracy/nb_eval_steps, eval_f1/nb_eval_steps, eval_recall/nb_eval_steps, eval_precesion/nb_eval_steps,)

    def fit_crossValidation(model, train_dataloader, validation_dataloader, epochs, device, optimizer, scheduler, criterion, writer=0, print_each=40):
        # Set the seed value all ovr the place to make this reproducible.
        seed_val = 2

        random.seed(seed_val)
        np.random.seed(seed_val)
        torch.manual_seed(seed_val)
        torch.cuda.manual_seed_all(seed_val)
        model_save_path = 'tmp'
        loss_values = []
        hist_valid_scores = []

        # For each epoch...
        for epoch_i in range(0, epochs):
            logs = {}

            # ========================================
            #               Training
            # ========================================

            # Perform one full pass over the training set.

            print("")
            print(
                '======== Epoch {:} / {:} ========'.format(epoch_i + 1, epochs))
            print('Training...')

            # Measure how long the training epoch takes.
            t0 = time.time()

            # Reset the total loss for this epoch.
            total_loss = 0
            total_accuracy = 0

            # Put the model into training mode. Don't be mislead--the call to
            # `train` just changes the *mode*, it doesn't *perform* the training.
            model.train()

            # For each batch of training data...
            for step, batch in enumerate(train_dataloader):

                # Progress update every 40 batches.
                if step % print_each == 0 and not step == 0:
                    # Calculate elapsed time in minutes.
                    elapsed = format_time(time.time() - t0)

                    # Report progress.
                    print('  Batch {:>5,}  of  {:>5,}.    Elapsed: {:}.'.format(
                        step, len(train_dataloader), elapsed))
                # move batch data to device (cpu or gpu)
                batch = tuple(t.to(device) for t in batch)

                # `batch` contains three pytorch tensors:
                #   [0]: input ids
                #   [1]: attention masks
                #   [2]: labels

                model.zero_grad()
                outputs = model(batch)

                # The call to `model` always returns a tuple, so we need to pull the
                # loss value out of the tuple.
                logits = outputs[0]
                label_ids = batch[-1]
                # print(logits)

                loss = criterion(logits.view(-1, model.n_class),
                                 label_ids.view(-1))

                # Move logits back to cpu for metrics calculations
                logits = logits.detach().cpu().numpy()

                # Calculate the accuracy for this batch of test sentences.
                current_accuracy = flat_accuracy(logits, label_ids)
                total_accuracy += current_accuracy
                # Accumulate the training loss over all of the batches so that we can
                # calculate the average loss at the end. `loss` is a Tensor containing a
                # single value; the `.item()` function just returns the Python value
                # from the tensor.
                total_loss += loss.item()
                writer.add_scalar('training loss',
                                  loss.item(),
                                  epoch_i * len(train_dataloader)+step)
                writer.add_scalar('training Accuracy',
                                  current_accuracy,
                                  epoch_i * len(train_dataloader)+step)

                # Perform a backward pass to calculate the gradients.
                loss.backward()

                # Clip the norm of the gradients to 1.0.
                # This is to help prevent the "exploding gradients" problem.
                torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)

                # Update parameters and take a step using the computed gradient.
                # The optimizer dictates the "update rule"--how the parameters are
                # modified based on their gradients, the learning rate, etc.
                optimizer.step()

                # Update the learning rate.
                scheduler.step()

            # Calculate the average loss over the training data.
            avg_train_loss = total_loss / len(train_dataloader)
            avg_train_accuracy = total_accuracy / len(train_dataloader)
            # Store the loss value for plotting the learning curve.
            loss_values.append(avg_train_loss)
            logs["log loss"] = avg_train_loss
            logs["accuracy"] = avg_train_accuracy

            print("")
            print("  Average training loss: {0:.2f}".format(avg_train_loss))
            print("  Training epcoh took: {:}".format(
                format_time(time.time() - t0)))

            # ========================================
            #               Validation
            # ========================================
            # After the completion of each training epoch, measure our performance on
            # our validation set.

            print("")
            print("Running Validation...")

            t0 = time.time()

            # Put the model in evaluation mode--the dropout layers behave differently
            # during evaluation.
            model.eval()

            # Tracking variables
            predictions, true_labels = [], []
            eval_loss, eval_accuracy, eval_f1, eval_recall, eval_precesion = 0, 0, 0, 0, 0
            nb_eval_steps, nb_eval_examples = 0, 0
            validation_loss = 0
            # Evaluate data for one epoch
            for step, batch in enumerate(validation_dataloader):

                # Add batch to GPU
                batch = tuple(t.to(device) for t in batch)

                # Unpack the inputs from our dataloader

                # Telling the model not to compute or store gradients, saving memory and
                # speeding up validation
                with torch.no_grad():

                    # Forward pass, calculate logit predictions.
                    # This will return the logits rather than the loss because we have
                    # not provided labels.
                    # token_type_ids is the same as the "segment ids", which
                    # differentiates sentence 1 and 2 in 2-sentence tasks.
                    # The documentation for this `model` function is here:
                    model.eval()
                    # https://huggingface.co/transformers/v2.2.0/model_doc/bert.html#transformers.BertForSequenceClassification
                    outputs = model(batch)

                # Get the "logits" output by the model. The "logits" are the output
                # values prior to applying an activation function like the softmax.

                logits = outputs[0]
                label_ids = batch[-1]
                predictions.extend(logits)
                true_labels.extend(label_ids)

                pred_flat = np.argmax(predictions, axis=1)
                true_labels = [dic_cat_labels.get(x) for x in true_labels]
                pred_flat = [dic_cat_labels.get(x) for x in pred_flat]

                cr = classification_report(true_labels, pred_flat, digits=4)
                print(accuracy_score(pred_flat_cat, true_labels_cat))
                print(cr)

                validation_loss += criterion(logits.view(-1,
                                                         model.n_class), label_ids.view(-1))

                # print(logits)

                # Move logits and labels to CPU
                logits = logits.detach().cpu().numpy()
                label_ids = label_ids.to('cpu').numpy()

                # Calculate the accuracy for this batch of test sentences.
                tmp_eval_accuracy = flat_accuracy(logits, label_ids)
                tmp_eval_f1 = flat_f1(logits, label_ids)
                tmp_eval_recall = flat_recall(logits, label_ids)
                tmp_eval_precision = flat_precision(logits, label_ids)
                # Accumulate the total scores.
                eval_accuracy += tmp_eval_accuracy
                eval_f1 += tmp_eval_f1
                eval_recall += tmp_eval_recall
                eval_precesion += tmp_eval_precision

                # Track the number of batches
                nb_eval_steps += 1
            validation_loss = validation_loss/len(validation_dataloader)
            is_better = len(hist_valid_scores) == 0 or validation_loss < min(
                hist_valid_scores)
            hist_valid_scores.append(validation_loss)

            if is_better:
                patience = 0
                print(
                    'save currently the best model to [%s]' % model_save_path, file=sys.stderr)
                model.save(model_save_path)
                # also save the optimizers' state
                torch.save(optimizer.state_dict(), model_save_path + '.optim')
            elif patience < 5:
                patience += 1
                '''print('hit patience %d' % patience, file=sys.stderr)
            if patience == int(5):
                # decay lr, and restore from previously best checkpoint
                print('load previously best model and decay learning rate to ', file=sys.stderr)
                # load model
                params = torch.load(model_save_path, map_location=lambda storage, loc: storage)
                model.load_state_dict(params['state_dict'])
                model = model.to(torch.device("cuda"))
                print('restore parameters of the optimizers', file=sys.stderr)
                optimizer.load_state_dict(torch.load(model_save_path + '.optim'))
                # reset patience
                patience = 0'''
            # Report the final accuracy for this validation run.
            print("  Accuracy: {0:.2f}".format(eval_accuracy/nb_eval_steps))
            print("  F1: {0:.2f}".format(eval_f1/nb_eval_steps))
            print("  Recall: {0:.2f}".format(eval_recall/nb_eval_steps))
            print("  Precision: {0:.2f}".format(eval_precesion/nb_eval_steps))
            print("  Validation took: {:}".format(
                format_time(time.time() - t0)))
        return (eval_accuracy/nb_eval_steps, eval_f1/nb_eval_steps, eval_recall/nb_eval_steps, eval_precesion/nb_eval_steps,)

        # load model

        # reset patience

    def fit_multitask(model, train_dataloader, validation_dataloader):
        # Set the seed value all ovr the place to make this reproducible.
        seed_val = 2

        random.seed(seed_val)
        np.random.seed(seed_val)
        torch.manual_seed(seed_val)
        torch.cuda.manual_seed_all(seed_val)
        model_save_path = 'tmp'
        loss_values = []
        hist_valid_scores = []

        # For each epoch...
        for epoch_i in range(0, epochs):
            logs = {}

            # ========================================
            #               Training
            # ========================================

            # Perform one full pass over the training set.

            print("")
            print(
                '======== Epoch {:} / {:} ========'.format(epoch_i + 1, epochs))
            print('Training...')

            # Measure how long the training epoch takes.
            t0 = time.time()

            # Reset the total loss for this epoch.
            total_loss = 0
            total_accuracy = 0

            # Put the model into training mode. Don't be mislead--the call to
            # `train` just changes the *mode*, it doesn't *perform* the training.
            model.train()

            # For each batch of training data...
            for step, batch in enumerate(train_dataloader):

                # Progress update every 40 batches.
                if step % 40 == 0 and not step == 0:
                    # Calculate elapsed time in minutes.
                    elapsed = format_time(time.time() - t0)

                    # Report progress.
                    print('  Batch {:>5,}  of  {:>5,}.    Elapsed: {:}.'.format(
                        step, len(train_dataloader), elapsed))

                # Unpack this training batch from our dataloader.
                #
                # As we unpack the batch, we'll also copy each tensor to the GPU using the
                # `to` method.
                #
                # `batch` contains three pytorch tensors:
                #   [0]: input ids
                #   [1]: attention masks
                #   [2:]: labels of different tasks

                # Always clear any previously calculated gradients before performing a
                # backward pass. PyTorch doesn't do this automatically because
                # accumulating the gradients is "convenient while training RNNs".
                # (source: https://stackoverflow.com/questions/48001598/why-do-we-need-to-call-zero-grad-in-pytorch)
                model.zero_grad()

                # Perform a forward pass (evaluate the model on this training batch).
                # This will return the loss (rather than the model output) because we
                # have provided the `labels`.
                # The documentation for this `model` function is here:
                # https://huggingface.co/transformers/v2.2.0/model_doc/bert.html#transformers.BertForSequenceClassification
                outputs = model(batch)

                # The call to `model` always returns a tuple, so we need to pull the
                # loss value out of the tuple.
                loss = outputs[0]
                logits = outputs[2]
                # print(logits)
                logits = logits.detach().cpu().numpy()
                # We monitore only the first task on training
                label_ids = batch[-1]

                # Calculate the accuracy for this batch of test sentences.
                current_accuracy = flat_accuracy(logits, label_ids)
                total_accuracy += current_accuracy
                # Accumulate the training loss over all of the batches so that we can
                # calculate the average loss at the end. `loss` is a Tensor containing a
                # single value; the `.item()` function just returns the Python value
                # from the tensor.
                total_loss += loss.item()
                writer.add_scalar('training loss',
                                  loss.item(),
                                  epoch_i * len(train_dataloader)+step)
                writer.add_scalar('training Accuracy',
                                  current_accuracy,
                                  epoch_i * len(train_dataloader)+step)

                # Perform a backward pass to calculate the gradients.
                loss.backward()

                # Clip the norm of the gradients to 1.0.
                # This is to help prevent the "exploding gradients" problem.
                torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)

                # Update parameters and take a step using the computed gradient.
                # The optimizer dictates the "update rule"--how the parameters are
                # modified based on their gradients, the learning rate, etc.
                optimizer.step()

                # Update the learning rate.
                scheduler.step()

            # Calculate the average loss over the training data.
            avg_train_loss = total_loss / len(train_dataloader)
            avg_train_accuracy = total_accuracy / len(train_dataloader)
            # Store the loss value for plotting the learning curve.
            loss_values.append(avg_train_loss)
            logs["log loss"] = avg_train_loss
            logs["accuracy"] = avg_train_accuracy

            print("")
            print("  Average training loss: {0:.2f}".format(avg_train_loss))
            print("  Training epcoh took: {:}".format(
                format_time(time.time() - t0)))

            # ========================================
            #               Validation
            # ========================================
            # After the completion of each training epoch, measure our performance on
            # our validation set.

            print("")
            print("Running Validation...")

            t0 = time.time()

            # Put the model in evaluation mode--the dropout layers behave differently
            # during evaluation.
            model.eval()

            # Tracking variables
            eval_loss, eval_accuracy, eval_f1, eval_recall, eval_precesion = 0, 0, 0, 0, 0
            nb_eval_steps, nb_eval_examples = 0, 0

            validation_loss = 0
            # Evaluate data for one epoch
            for batch in validation_dataloader:

                # Add batch to GPU
                batch = tuple(t.to(torch.device('cuda')) for t in batch)

                # Unpack the inputs from our dataloader
                b_input_ids, b_input_mask, b_meta_features, b_labels_cat = batch

                # Telling the model not to compute or store gradients, saving memory and
                # speeding up validation
                with torch.no_grad():

                    # Forward pass, calculate logit predictions.
                    # This will return the logits rather than the loss because we have
                    # not provided labels.
                    # token_type_ids is the same as the "segment ids", which
                    # differentiates sentence 1 and 2 in 2-sentence tasks.
                    # The documentation for this `model` function is here:
                    model.eval()
                    # https://huggingface.co/transformers/v2.2.0/model_doc/bert.html#transformers.BertForSequenceClassification
                    outputs = model(batch)

                # Get the "logits" output by the model. The "logits" are the output
                # values prior to applying an activation function like the softmax.
                validation_loss += outputs[0]
                logits = outputs[2]
                # print(logits)

                # Move logits and labels to CPU
                logits = logits.detach().cpu().numpy()
                label_ids = b_labels_cat.to('cpu').numpy()

                # Calculate the accuracy for this batch of test sentences.
                tmp_eval_accuracy = flat_accuracy(logits, label_ids)
                tmp_eval_f1 = flat_f1(logits, label_ids)
                tmp_eval_recall = flat_recall(logits, label_ids)
                tmp_eval_precision = flat_precision(logits, label_ids)
                # Accumulate the total scores.
                eval_accuracy += tmp_eval_accuracy
                eval_f1 += tmp_eval_f1
                eval_recall += tmp_eval_recall
                eval_precesion += tmp_eval_precision

                # Track the number of batches
                nb_eval_steps += 1
            validation_loss = validation_loss/len(validation_dataloader)
            is_better = len(hist_valid_scores) == 0 or validation_loss < min(
                hist_valid_scores)
            hist_valid_scores.append(validation_loss)

            if is_better:
                patience = 0
                print(
                    'save currently the best model to [%s]' % model_save_path, file=sys.stderr)
                model.save(model_save_path)
                # also save the optimizers' state
                torch.save(optimizer.state_dict(), model_save_path + '.optim')
            elif patience < 5:
                patience += 1
                '''print('hit patience %d' % patience, file=sys.stderr)
            if patience == int(5):
                # decay lr, and restore from previously best checkpoint
                print('load previously best model and decay learning rate to ', file=sys.stderr)
                # load model
                params = torch.load(model_save_path, map_location=lambda storage, loc: storage)
                model.load_state_dict(params['state_dict'])
                model = model.to(torch.device("cuda"))
                print('restore parameters of the optimizers', file=sys.stderr)
                optimizer.load_state_dict(torch.load(model_save_path + '.optim'))
                # reset patience
                patience = 0'''
            # Report the final accuracy for this validation run.
            print("  Accuracy: {0:.2f}".format(eval_accuracy/nb_eval_steps))
            print("  F1: {0:.2f}".format(eval_f1/nb_eval_steps))
            print("  Recall: {0:.2f}".format(eval_recall/nb_eval_steps))
            print("  Precision: {0:.2f}".format(eval_precesion/nb_eval_steps))
            print("  Validation took: {:}".format(
                format_time(time.time() - t0)))

        # load model

        # reset patience
