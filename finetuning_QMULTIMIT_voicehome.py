import os
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.loggers import WandbLogger
import wandb
from model import SeldTrainer
import pandas as pd
from QMULTIMIT import QMULLIMITDataModule

if __name__ == "__main__":
    # QMULTIMIT
    data_dir = '/mnt/data3/SDE'
    path_audios = os.path.join(data_dir, 'real_datasets/QMULTIMIT' )    
    pathNoisesTraining = os.path.join(data_dir, "noise_dataset/training_noise")
    pathNoisesVal = os.path.join(data_dir, "noise_dataset/val_noise")
    pathNoisesTest = os.path.join(data_dir, "noise_dataset/test_noise")

    path_audio1 = os.path.join(data_dir, 'real_datasets/voiceHome2-2_corpus_1.0')
    # pathNoisesTraining = os.path.join(data_dir, "noise_dataset/training_noise")
    # pathNoisesVal = os.path.join(data_dir, "noise_dataset/val_noise")
    # pathNoisesTest = os.path.join(data_dir, "noise_dataset/test_noise")

    # FIXED PARAMS
    # config = {
    #     "max_epochs": 50,
    #     "batch_size": 16,
    #     "lr": 0.001,
    #     "sampling_frequency": 16000,
    #     "dBNoise" : [30, 20 , 10, 0, -10],
    #     "kernels": ["freq"],
    #     "n_grus": [2],
    #     "features_set": ["all"],
    #     "att_conf": ["Nothing", "onSpec"]
    # }
    config = {
        "max_epochs": 50,
        "batch_size": 16,
        "lr": 0.001,
        "sampling_frequency": 16000,
        "dBNoise" : [None],
        "kernels": ["freq"],
        "n_grus": [2],
        "features_set": ["all"],
        "att_conf": ["Nothing", "onSpec"]
    }
    for db in config['dBNoise']:
        for n_gru in config['n_grus']:
            for kernel in config['kernels']:
                for features in config['features_set']:
                    for att_conf in config['att_conf']:
                        seed_everything(42) # workers=True
                        # all_results = pd.DataFrame([], columns = ["GT", "Pred", "L1", "rL1", "ID"])
                        run_name = "Kernels{}_Gru{}_Features_{}Att_conf{}_QMULTIMIT_{}dB".format(kernel, n_gru, features, att_conf, db)
                        model = SeldTrainer(lr=config["lr"], kernels = kernel, n_grus = n_gru, features_set = features, att_conf = att_conf)
                        datamodule = QMULLIMITDataModule(path_dataset = path_audios, batch_size = config["batch_size"], pathNoisesTraining=pathNoisesTraining,
                                        pathNoisesVal=pathNoisesVal, pathNoisesTest=pathNoisesTest, db = db)
                        wandb_logger = WandbLogger(
                                        project="Distance-Estimation-RQ1",
                                        name="{} Noisy {}".format(run_name, db),
                                        tags=["TABLE4", "Hybrid"],
                                    )
                        trainer = Trainer(
                                            accelerator="gpu",
                                            devices = [6],
                                            log_every_n_steps = 50,
                                            max_epochs=config["max_epochs"],
                                            precision = 32,
                                            logger=wandb_logger,
                                            deterministic=True
                                        )
                        wandb_logger.log_hyperparams(config)
                        wandb_logger.watch(model, log_graph=False)
                        trainer.fit(model, datamodule)
                        trainer.test(model, datamodule)
                        wandb.finish()
                        all_results = pd.DataFrame(model.all_test_results)
                        all_results.to_csv(run_name + ".csv")
