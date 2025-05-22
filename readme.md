**Step1**

Unzip final_data, and put data (something like redial_rec) into the /data fold.
**Step2**

sh train_pre(_inspired).sh (UniCRS only)

sh train_rec_pretrain(_inspired).sh ## pretrain on synthetic dataset

sh train_rec_ft(_inspired).sh ## finetune on real dataset

**Note** 
Please refer to the results in the final_evaluation (started with "final_evaluation:evaluate" in the logger file, printed after the training process is finished). It will filter out repeated items in the prediction list. The results printed during training do not filter out repeated items in the prediction list.

You can also run final_evaluation.py to get results using saved checkpoints.

If you want to run base models (without data augmentation), you can run:

sh train_pre(_inspired).sh (UniCRS only)

sh train_rec_base(_inspired).sh

**Acknowledge**
data and code is based on UniCRS[1] and CFCRS[2]

[1] Towards Unified Conversational Recommender Systems via Knowledge-Enhanced Prompt Learning

[2] Improving Conversational Recommendation Systems via Counterfactual Data Simulation