import numpy as np


class PerformanceMetrics:
    def accuracy(self, y_pred, y_true):
        predictions = np.argmax(y_pred, axis=1)

        # if len(y_true.shape) == 1:
        # we can keep the same y_true

        """
        Let predictions were -
            0.1 0.6 0.3
            0.7 0.1 0.2
            0.2 0.3 0.5
        
        Argmax will collapse them to-
            1
            0
            2
        
        Now let the y_true for each sample in batch was-
            1
            1
            2
        
        We can now compare them

        Similarly for one-hot-encoded values-

        let y_true was-
            0 1 0
            1 0 0
            0 0 1

        Argmax will collpase it to
            1
            0
            2
        
        These can now be compared to already collpased y_pred values

        """

        # For one hot encoded values-
        if len(y_true.shape) == 2:
            y_true = np.argmax(y_true, axis=1)

        accuracy = np.mean(predictions == y_true)

        # For one hot encoded values:
        # if len(y_true.shape) == 2:
        #     pass

        return accuracy
