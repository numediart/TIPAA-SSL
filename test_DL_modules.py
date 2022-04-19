from DL_accuracy_performance import *

def test_pConstrast():
    pContrast_for_actor_recordings(target_phones='AO1')
def test_termination_contrast():
    termination_contrast_from_audiobook_data(data_set='test-other', target_phones='D', n=50)
    termination_contrast_for_actor_recordings()

def a_test_stress_detection():
    stress_GE_performance_test(level='word')