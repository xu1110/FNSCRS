**Step1**

Unzip final_data, and put data (something like redial_rec, inspired_rec) into the /data fold of the corresponding model, which include both the original dataset({train/valid/test}_data_processed.jsonl) and the augmented dataset (train_data_aug_gritlm7b_reward_model_50.jsonl).

**Step2**

sh train_pre(_inspired).sh (UniCRS only)

sh train_rec_pretrain(_inspired).sh ## pretrain on the synthetic dataset

sh train_rec_ft(_inspired).sh ## finetune on the real dataset, you can use --kl_coef to add soft label.

**Note** 

Please refer to the results in the final_evaluation (started with "final_evaluation:evaluate" in the logger file, printed after the training process is finished). It will filter out repeated items in the prediction list. The results printed during training do not filter out repeated items in the prediction list.

You can also run final_evaluation.py to get results using saved checkpoints.

If you want to run base models (without data augmentation), you can run:

sh train_pre(_inspired).sh (UniCRS only)

sh train_rec_base(_inspired).sh

**Acknowledge**

Data and code is based on CFCRS[1] and UniCRS[2]. Please refer to CFCRS for raw data processing.

[1] Improving Conversational Recommendation Systems via Counterfactual Data Simulation

[2] Towards Unified Conversational Recommender Systems via Knowledge-Enhanced Prompt Learning

**Contact**

My email address is xuhaozhe1110@gmail.com
