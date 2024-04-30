package: 

hydra-core
git+https://github.com/JudyYe/mpi_utils.git




bash scripts/shape_color_mv.sh human/yuliang_noattn xiu human/yuliang


# *puzzlebooth* MVFusion by xiu
bash multi_concept_mv/train_puzzlebooth.sh \
        data/human/yuliang/ \
        output/puzzle_int_noattn \
        traintest mv \

 
# *dreambooth* MVFusion by xiu 
python -m multi_concept_mv.train_dreambooth_diffuser \
    report_to=wandb \
    expname= \



# ---------------------------
bash multi_concept_mv/test_puzzlebooth.sh \
    data/human/yuliang/ \
    output/dev_puzzle \
    tmp mv \




python -m mpi_utils.mpi_wrapper --sl_time 1 --sl_cuda 8.5 \
    bash multi_concept_mv/train_puzzlebooth.sh \
        data/human/yuliang/ \
        output/puzzle_remote \
        traintest mv \

