import os
max_iteration = 5
is_done_extract = False
is_done_auto_annot = False
is_done_correction = False
is_on_correction = False
is_done_train = False
frame_index = 0
reject_index = 0
epochs_per_iteration = 10
epochs_first_training = 10
real_false_positive = 0
real_false_negative = 0
is_include_reject_in_auto_annot = True

# 0 = DETETION mode, 1 = ADDITION mode
manual_corrections_mode = 0

# DELETION mode
# 0 = DISPLAY all, 1 = DISPLAY per object
deletion_display_mode = 0
# number of skipping command
deletion_skip_num = 5
# number of deletion
deletion_count = 0

#ADDITION mode
addition_skip_num = 5
# number of addition
addition_count = 0


# Relative path: ../../self-training (naik 2 level dari app/data ke workv1)
working_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../self-training"))
evaluation_summary_dir = None
video_path = None
is_random = False
model_path = None
eval_path = None
frame_dir = None
iteration = 1

first_training_dataset_dir = None

done_IBBA = False
train_awal_dir = None

def reset_data():
    global max_iteration, iteration
    global is_done_extract, is_done_auto_annot, is_done_train, is_done_correction, is_on_correction
    global frame_index, reject_index
    global epochs_per_iteration, epochs_first_training
    global real_false_positive, real_false_negative
    global manual_corrections_mode
    global deletion_display_mode, deletion_skip_num, deletion_count
    global addition_skip_num, addition_count
    global evaluation_summary_dir, video_path, is_random
    global model_path, eval_path, frame_dir
    global first_training_dataset_dir
    global working_dir
    global is_include_reject_in_auto_annot
    global done_IBBA, train_awal_dir

    max_iteration = 5
    iteration = 1

    is_done_extract = False
    is_done_auto_annot = False
    is_done_correction = False
    is_on_correction = False
    is_done_train = False
    is_include_reject_in_auto_annot = True

    frame_index = 0
    reject_index = 0

    epochs_per_iteration = 10
    epochs_first_training = 10

    real_false_positive = 0
    real_false_negative = 0

    manual_corrections_mode = 0
    deletion_display_mode = 0
    deletion_skip_num = 5
    deletion_count = 0

    addition_skip_num = 5
    addition_count = 0

    # path
    working_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../self-training"))
    evaluation_summary_dir = None
    video_path = None
    is_random = False
    model_path = None
    eval_path = None
    frame_dir = None
    first_training_dataset_dir = None

    done_IBBA = False
    train_awal_dir = None
