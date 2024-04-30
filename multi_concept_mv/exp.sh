
bash scripts/shape_color_mv.sh human/yuliang_500 xiu human/yuliang


 
bash multi_concept_mv/test_puzzlebooth.sh \
    data/human/yuliang/ \
    output/dev_puzzle \
    tmp mv \


# puzzlebooth MVFusion by xiu

bash multi_concept_mv/train_puzzlebooth.sh \
        data/human/yuliang/ \
        output/puzzle_int_noattn \
        traintest mv \



python -m mpi_utils.mpi_wrapper --sl_time 1 --sl_cuda 8.5 \
    bash multi_concept_mv/train_puzzlebooth.sh \
        data/human/yuliang/ \
        output/puzzle_remote \
        traintest mv \


# dreambooth MVFusion by xiu 
python -m multi_concept_mv.train_dreambooth_diffuser \
    report_to=wandb