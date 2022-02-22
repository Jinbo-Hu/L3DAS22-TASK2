from utils.common import find_key_from_value
class BaseInferer:
    """ Base inferer class

    """
    def infer(self, *args, **kwargs):
        """ Perform an inference on test data.

        """
        raise NotImplementedError

    def fusion(self, submissions_dir, preds):
        """ Ensamble predictions.

        """
        raise NotImplementedError        

    @staticmethod
    def write_submission(label_dic_task2, submissions_dir, pred_list):
        """ Write predicted result to submission csv files
        Args:
            pred_dict: L3DASS22 format list:
                pred_dict[frame-containing-events] = [[event1, x, y, z],[event2, x, y, z],[event3, x, y, z]]
        """
        num_frames = len(pred_list)
        frame_label = pred_list
        for frame in range(num_frames):
            if frame_label[frame]:
                for event in frame_label[frame]:
                    event[0] = find_key_from_value(label_dic_task2, event[0])[0]
                    with submissions_dir.open('a') as f:
                        f.write('{},{},{},{},{}\n'.format(frame, event[0], event[1], event[2], event[3]))   
        



