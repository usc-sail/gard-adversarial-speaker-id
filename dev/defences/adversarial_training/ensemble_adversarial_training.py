"""
This module implements emsemble adversarial training that can use multiple attack methods like PGD, FGSM together 
"""

from __future__ import absolute_import, division, print_function, unicode_literals

import logging

from art.defences.trainer import AdversarialTrainer
from art.data_generators import PyTorchDataGenerator
import numpy as np
import pdb
import torch
import copy

logger = logging.getLogger(__name__)

class EnsembleAdversarialTrainer():
    def __init__(self, classifier, attack_methods, ratio=0.5, augment=0):
        #self.adversarial_trainer = AdversarialTrainer(classifier=classifier, attacks=attack_methods, ratio=ratio)
        self.adversarial_trainer = AdversarialTrainerWrapper(classifier=classifier, attacks=attack_methods, ratio=ratio, augment=augment)
    
    def fit_generator(self, train_generator, epochs):
        train_generator_art = PyTorchDataGenerator(train_generator, len(train_generator.dataset), train_generator.batch_size)
        self.adversarial_trainer.fit_generator(train_generator_art, nb_epochs=epochs) 

    def get_backend_model(self):
        return self.adversarial_trainer.classifier._model._model 
            
    def get_backend_optimizer(self):
        return self.adversarial_trainer.classifier._optimizer 
     

class AdversarialTrainerWrapper(AdversarialTrainer):
    def __init__(self, classifier, attacks, ratio=0.5, augment=0):
        super().__init__(classifier, attacks, ratio=0.5)
        self.augment = augment

    def fit_generator(self, generator, nb_epochs=20, **kwargs):
        """
        Train a model adversarially using a data generator.
        See class documentation for more information on the exact procedure.

        :param generator: Data generator.
        :type generator: :class:`.DataGenerator`
        :param nb_epochs: Number of epochs to use for trainings.
        :type nb_epochs: `int`
        :param kwargs: Dictionary of framework-specific arguments. These will be passed as such to the `fit` function of
               the target classifier.
        :type kwargs: `dict`
        :return: `None`
        """
        logger.info("Performing adversarial training using %i attacks.", len(self.attacks))
        size = generator.size
        batch_size = generator.batch_size
        nb_batches = int(np.ceil(size / batch_size))
        ind = np.arange(generator.size)
        attack_id = 0

        # Precompute adversarial samples for transferred attacks
        logged = False
        self._precomputed_adv_samples = []
        for attack in self.attacks:
            if "targeted" in attack.attack_params:
                if attack.targeted:
                    raise NotImplementedError("Adversarial training with targeted attacks is currently not implemented")

            if attack.classifier != self.classifier:
                if not logged:
                    logger.info("Precomputing transferred adversarial samples.")
                    logged = True

                next_precomputed_adv_samples = None
                for batch_id in range(nb_batches):
                    # Create batch data
                    x_batch, y_batch = generator.get_batch()
                    x_adv_batch = attack.generate(x_batch, y=y_batch)
                    if next_precomputed_adv_samples is None:
                        next_precomputed_adv_samples = x_adv_batch
                    else:
                        next_precomputed_adv_samples = np.append(next_precomputed_adv_samples, x_adv_batch, axis=0)
                self._precomputed_adv_samples.append(next_precomputed_adv_samples)
            else:
                self._precomputed_adv_samples.append(None)

        for i_epoch in range(nb_epochs):
            logger.info("Adversarial training epoch %i/%i", i_epoch, nb_epochs)

            # Shuffle the indices of precomputed examples
            np.random.shuffle(ind)

            all_accuracies = []
            all_accuracies_normal = []
            all_accuracies_adv = []
            for batch_id in range(nb_batches):
                # Create batch data
                x_batch, y_batch = generator.get_batch()
                x_batch = x_batch.copy()
                #x_batch_DEBUG = copy.deepcopy(x_batch)

                # Choose indices to replace with adversarial samples
                nb_adv = int(np.ceil(self.ratio * x_batch.shape[0]))
                attack = self.attacks[attack_id]
                if self.ratio < 1:
                    adv_ids = np.random.choice(x_batch.shape[0], size=nb_adv, replace=False)
                else:
                    adv_ids = list(range(x_batch.shape[0]))
                    np.random.shuffle(adv_ids)

                # If source and target models are the same, craft fresh adversarial samples
                if attack.classifier == self.classifier:
                    x_batch[adv_ids] = attack.generate(x_batch[adv_ids], y=y_batch[adv_ids])

                    #print("Max pertrubations on adversarial samples")
                    #delta_ = np.max(x_batch_DEBUG[adv_ids, ...] - x_batch[adv_ids, ...], 2)
                    #print(delta_)
                    

                # Otherwise, use precomputed adversarial samples
                else:
                    x_adv = self._precomputed_adv_samples[attack_id]
                    x_adv = x_adv[ind[batch_id * batch_size : min((batch_id + 1) * batch_size, size)]][adv_ids]
                    x_batch[adv_ids] = x_adv

                #Gaussian augmentation to normal samples
                all_ids = range(x_batch.shape[0])
                normal_ids = [i_ for i_ in all_ids if i_ not in adv_ids]
                if self.augment==1:
                    x_batch_normal = x_batch[normal_ids, ...]
                    y_batch_normal = y_batch[normal_ids, ...]

                    a = np.random.rand()
                    noise = a * 0.008 * np.random.rand(*x_batch_normal.shape)                    
                    #add noise
                    x_batch_normal_noisy = x_batch_normal + noise.astype('float32')
                    x_batch = np.vstack((x_batch, x_batch_normal_noisy))
                    y_batch = np.concatenate((y_batch, y_batch_normal))



                # Fit batch
                #JATI--start
                self.classifier.set_learning_phase(True)
                #JATI--end
                self.classifier.fit(x_batch, y_batch, nb_epochs=1, batch_size=x_batch.shape[0], **kwargs)
                attack_id = (attack_id + 1) % len(self.attacks) 

                #calculate training accuracy
                 #JATI--start
                self.classifier.set_learning_phase(False)
                #JATI--end
                predictions = self.classifier.predict(x_batch)
                acc = np.mean(predictions.argmax(1)==y_batch)
                predictions_adv = predictions[adv_ids]
                acc_adv = np.mean(predictions_adv.argmax(1) == y_batch[adv_ids])
                #all_ids = range(x_batch.shape[0])
                #normal_ids = [i_ for i_ in all_ids if i_ not in adv_ids]
                predictions_normal = predictions[normal_ids]
                acc_normal = np.mean(predictions_normal.argmax(1)==y_batch[normal_ids])

                print("Batch", batch_id, "/", nb_batches, ": Acc = ", round(acc,4), "\tAcc adv =", round(acc_adv,4), "\tAcc normal =", round(acc_normal,4))
                logger.info("Batch {}/{}: Acc = {:.6f}\tAcc adv = {:.6f}\tAcc normal = {:.6f}".format(batch_id, nb_batches, acc, acc_adv, acc_normal))

                all_accuracies.append(acc)
                all_accuracies_normal.append(acc_normal)
                all_accuracies_adv.append(acc_adv)

            print()
            print('--------------------------------------')
            print("EPOCH", i_epoch, "/", nb_epochs, ": Acc = ", round(np.mean(all_accuracies),4), "\tAcc adv =", round(np.mean(all_accuracies_adv),4), "\tAcc normal =", round(np.mean(all_accuracies_normal),4))
            print('--------------------------------------')
            print()
            logger.info("EPOCH {}/{}: Acc = {:.6f}\tAcc adv = {:.6f}\tAcc normal = {:.6f}".format(i_epoch, nb_epochs, np.mean(all_accuracies), np.mean(all_accuracies_adv), np.mean(all_accuracies_normal)))


    
        
