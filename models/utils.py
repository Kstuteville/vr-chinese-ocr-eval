def build_baseline(input_shape):
  """
  Build a binary classification baseline using Dense layers.

  Parameters
  ----------
  input_shape : int
      The shape of the input data. This should be an integer
      specifying the number of features in the input data (e.g. 200).

  Returns
  -------
  baseline : keras.models.Sequential
      A Keras sequential model object representing the built model.
      The model architecture should consist of at least two Dense layers with
      appropriate sizes and activations, followed by an output layer with a 
      single unit and sigmoid activation function. The model should be
      compiled with the binary cross-entropy loss function, an adam optimizer, 
      and the accuracy metric.
  """
  # YOUR CODE HERE
  # Build the model


def train_model(model, X_train, y_train, epochs=10, batch_size=32, \
                class_weights=None, X_val=None, y_val=None):
    """
    Train a given Keras model using the provided training data. Use the 
    validation data to monitor the model during training.

    Parameters
    ----------
    model : keras.Sequential
        The Keras model to be trained.
    X_train : np.ndarray
        The input features for training the model.
    y_train : np.ndarray
        The target values for training the model.
    epochs : int, optional
        The number of epochs to train the model for. Default is 10.
    batch_size : int, optional
        The batch size to use during training. Default is 32.
    class_weights : dict, optional
        A dictionary containing class weights to be applied to the loss function
        during training to balance the data. Default is None.
    X_val : np.ndarray, optional
        The input features for validating the model.
    y_val : np.ndarray, optional
        The target values for validating the model.

    Returns
    -------
    model : keras.Sequential
        The trained Keras model.
    history : keras.callbacks.History
        The training history of the model.
    """
    # YOUR CODE HERE
    # Train the model
    history = model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, 
              class_weight=class_weights, validation_data=(X_val, y_val))
    return model, history
