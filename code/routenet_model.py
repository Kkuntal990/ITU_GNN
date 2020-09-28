"""
   Copyright 2020 Universitat Politècnica de Catalunya & AGH University of Science and Technology

                                        BSD 3-Clause License

   Redistribution and use in source and binary forms, with or without modification, are permitted
   provided that the following conditions are met:
    1. Redistributions of source code must retain the above copyright notice, this list of conditions
       and the following disclaimer.
    2. Redistributions in binary form must reproduce the above copyright notice, this list of
       conditions and the following disclaimer in the documentation and/or other materials provided
       with the distribution.
    3. Neither the name of the copyright holder nor the names of its contributors may be used to endorse
       or promote products derived from this software without specific prior written permission.

    THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED
    WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A
    PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR
    ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED
    TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION)
    HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
    NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
    POSSIBILITY OF SUCH DAMAGE.
"""

from __future__ import print_function
import tensorflow as tf

from spektral.layers import GraphAttention
POLICIES = {'WFQ': 0, 'SP': 1, 'DRR': 2}



class RouteNetModel(tf.keras.Model):
    """ Init method for the custom model.

    Args:
        config (dict): Python dictionary containing the diferent configurations
                       and hyperparameters.
        output_units (int): Output units for the last readout's layer.

    Attributes:
        config (dict): Python dictionary containing the diferent configurations
                       and hyperparameters.
        link_update (GRUCell): Link GRU Cell used in the Message Passing step.
        path_update (GRUCell): Path GRU Cell used in the Message Passing step.
        readout (Keras Model): Readout Neural Network. It expects as input the
                               path states and outputs the per-path delay.
    """

    def __init__(self, config, output_units=1):
        super(RouteNetModel, self).__init__()

        # Configuration dictionary. It contains the needed Hyperparameters for the model.
        # All the Hyperparameters can be found in the config.ini file
        self.config = config

        # GRU Cells used in the Message Passing step
        self.link_update = tf.keras.layers.GRUCell(
            int(self.config['HYPERPARAMETERS']['link_state_dim']))
        self.path_update = tf.keras.layers.GRUCell(
            int(self.config['HYPERPARAMETERS']['path_state_dim']))
        self.b_path_udpate = tf.keras.layers.GRUCell(
            int(self.config['HYPERPARAMETERS']['path_state_dim']))
        self.node_update = tf.keras.layers.GRUCell(
            int(self.config['HYPERPARAMETERS']['node_state_dim']))

        # Readout Neural Network. It expects as input the path states and outputs the per-path delay
        self.GAT = GraphAttention(channels = int(self.config['HYPERPARAMETERS']['node_state_dim']), 
                                  attn_heads=24 ,concat_heads=False)
        self.readout = tf.keras.Sequential([
            tf.keras.layers.Input(
                shape=int(self.config['HYPERPARAMETERS']['path_state_dim'])),
            tf.keras.layers.Dense(int(self.config['HYPERPARAMETERS']['readout_units']),
                                  activation=tf.nn.selu,
                                  kernel_regularizer=tf.keras.regularizers.l2(
                                      float(self.config['HYPERPARAMETERS']['l2'])),
                                  ),


            tf.keras.layers.Dense(int(self.config['HYPERPARAMETERS']['readout_units']),
                                  activation=tf.nn.relu,
                                  kernel_regularizer=tf.keras.regularizers.l2(
                                      float(self.config['HYPERPARAMETERS']['l2']))),

            tf.keras.layers.Dense(output_units,
                                  kernel_regularizer=tf.keras.regularizers.l2(
                                      float(self.config['HYPERPARAMETERS']['l2_2'])))

        ])
        #FIXME: Warning , model was inititated for (none, 32) but this operation is on (None, None, 32)

        '''self.merge = tf.keras.Sequential([
            #input = [bs, src_len,path_state+backstate]
            tf.keras.layers.Input(
                shape=2*int(self.config['HYPERPARAMETERS']['path_state_dim'])),
            tf.keras.layers.Dense(int(self.config['HYPERPARAMETERS']['path_state_dim']))
        ])

        self.attention = tf.keras.Sequential([
            #input = [bs, src_len,path_state+backstate]
            tf.keras.layers.Input(
                shape=(None,int(self.config['HYPERPARAMETERS']['link_state_dim']))),
            tf.keras.layers.Dense(1)
        ])'''


    def call(self, inputs, training=False):
        """This function is execution each time the model is called

        Args:
            inputs (dict): Features used to make the predictions.
            training (bool): Whether the model is training or not. If False, the
                             model does not update the weights.

        Returns:
            tensor: A tensor containing the per-path delay.
        """

        f_ = inputs
        #links = f_['links']
        paths = f_['paths']
        seqs = f_['sequences']
        ToS = f_['ToS']
        q_policy = f_['Q_policy']
        w_1 = f_['w1']
        w_2 = f_['w2']
        w_3 = f_['w3']
        nodes = f_['node_indices']
        queue_sizes = f_['queue_size']

        """
        Let's use type of queing policy of a node as 0,1,2 as {'WFQ', 'SP', 'DRR'},
        and also insert weight of each queue of the node.

        """
        shape = shape = tf.stack([
            f_['n_nodes'],
            int(self.config['HYPERPARAMETERS']['node_state_dim']) - 7
        ], axis=0)

        node_state = tf.concat([
            tf.expand_dims(q_policy, axis=1),
            tf.expand_dims(w_1, axis=1),
            tf.expand_dims(w_2, axis=1),
            tf.expand_dims(w_3, axis=1),
            queue_sizes,
            tf.zeros(shape)
        ], axis=1)

        # Compute the shape for the  all-zero tensor for link_state
        shape = tf.stack([
            f_['n_links'],
            int(self.config['HYPERPARAMETERS']['link_state_dim']) - 1
        ], axis=0)

        # Initialize the initial hidden state for links
        link_state = tf.concat([
            tf.expand_dims(f_['link_capacity'], axis=1),
            tf.zeros(shape,)
        ], axis=1)

        # Compute the shape for the  all-zero tensor for path_state
        shape = tf.stack([
            f_['n_paths'],
            int(self.config['HYPERPARAMETERS']['path_state_dim']) - 2
        ], axis=0)

        # Initialize the initial hidden state for paths
        path_state = tf.concat([
            tf.expand_dims(f_['bandwith'], axis=1),
            tf.expand_dims(ToS, axis=1),
            tf.zeros(shape)
        ], axis=1)

        ids = tf.stack([paths, seqs], axis=1)
        max_len = tf.reduce_max(seqs) + 1
        links1 = f_["adj"]
        links1 = tf.reshape(links1,(-1,1))
        adj = tf.zeros((f_['n_nodes']**2))
        adj = tf.tensor_scatter_nd_update(adj, links1, link_state[:,0])
        adj = tf.reshape(adj,(f_['n_nodes'],f_['n_nodes']))
        adj = tf.linalg.normalize(adj)[0]

        

        # Iterate t times doing the message passing
        for _ in range(int(self.config['HYPERPARAMETERS']['t'])):

            node_state = self.GAT([node_state ,adj ])
            h_tild = tf.gather(node_state, nodes)
            shape = tf.stack([
                f_['n_paths'],
                max_len,
                int(self.config['HYPERPARAMETERS']['node_state_dim'])])
            lens = tf.math.segment_sum(data=tf.ones_like(paths),
                                       segment_ids=paths)
            # Generate the aforementioned tensor [n_paths, max_len_path, dimension_node]
            node_inputs = tf.scatter_nd(ids, h_tild, shape)

            # The following lines generate a tensor of dimensions [n_paths, max_len_path, dimension_link] with all 0
            # but the link hidden states
            # h_tild = tf.gather(link_state, links)

            # shape = tf.stack([
            #     f_['n_paths'],
            #     max_len,
            #     int(self.config['HYPERPARAMETERS']['link_state_dim'])])

            # # Generate the aforementioned tensor [n_paths, max_len_path, dimension_link]
            # link_inputs = tf.scatter_nd(ids, h_tild, shape)

            #attn = self.attention(link_inputs)

            #link_inputs = tf.multiply(link_inputs, attn)

            # Define the RNN used for the message passing links to paths
            forward = tf.keras.layers.RNN(self.path_update,
                                          return_sequences=True,
                                          return_state=True)
            backward = tf.keras.layers.RNN(self.b_path_udpate,
                                           return_sequences=True,
                                           return_state=True, go_backwards=True)

            gru_rnn = tf.keras.layers.Bidirectional(
                forward, backward_layer=backward,  merge_mode="sum")
            

            # First message passing: update the path_state
            # outputs, path_state, b_path_state = gru_rnn(inputs=link_inputs,
            #                                             initial_state=[path_state,path_state],
            #                               mask=tf.sequence_mask(lens))

            # #path_state = self.merge(tf.concat([path_state, b_path_state] ,axis=1))
            # # For every link, gather and sum the sequence of hidden states of the paths that contain it
            # m = tf.gather_nd(outputs, ids)
            # m = tf.math.unsorted_segment_sum(m, links, f_['n_links'])

            outputs, path_state, b_path_state = gru_rnn(inputs=node_inputs,
                                                        initial_state=[path_state, path_state],
                                                        mask=tf.sequence_mask(lens))
            #path_state = self.merge(tf.concat([path_state, b_path_state] ,axis=1))

            m2 = tf.gather_nd(outputs, ids)
            m2 = tf.math.unsorted_segment_sum(m2, nodes, f_['n_nodes'])
            # Second message passing: update the link_state
            
            node_state, _ = self.node_update(m2, [node_state])
            

        # Call the readout ANN and return its predictions
        # print(path_state.shape)

        # ww = tf.gather(node_state, nodes);
        r = self.readout(path_state, training=training)
        # print(r.shape)
        return r
        


def r_squared(labels, predictions):
    """Computes the R^2 score.

        Args:
            labels (tf.Tensor): True values
            labels (tf.Tensor): This is the second item returned from the input_fn passed to train, evaluate, and predict.
                                If mode is tf.estimator.ModeKeys.PREDICT, labels=None will be passed.

        Returns:
            tf.Tensor: Mean R^2
        """

    total_error = tf.reduce_sum(tf.square(labels - tf.reduce_mean(labels)))
    unexplained_error = tf.reduce_sum(tf.square(labels - predictions))
    r_sq = 1.0 - tf.truediv(unexplained_error, total_error)

    # Needed for tf2 compatibility.
    m_r_sq, update_rsq_op = tf.compat.v1.metrics.mean(r_sq)

    return m_r_sq, update_rsq_op


def model_fn(features, labels, mode, params):
    """model_fn used by the estimator, which, given inputs and a number of other parameters,
       returns the ops necessary to perform training, evaluation, or predictions.

    Args:
        features (dict): This is the first item returned from the input_fn passed to train, evaluate, and predict.
        labels (tf.Tensor): This is the second item returned from the input_fn passed to train, evaluate, and predict.
                            If mode is tf.estimator.ModeKeys.PREDICT, labels=None will be passed.
        mode (tf.estimator.ModeKeys): Specifies if this is training, evaluation or prediction.
        params (dict): Dict of hyperparameters. Will receive what is passed to Estimator in params parameter.

    Returns:
        tf.estimator.EstimatorSpec: Ops and objects returned from a model_fn and passed to an Estimator.
    """

    # Create the model.
    model = RouteNetModel(params)

    # Execute the call function and obtain the predictions.
    predictions = model(features, training=(
        mode == tf.estimator.ModeKeys.TRAIN))

    predictions = tf.squeeze(predictions)

    # If we are performing predictions.
    if mode == tf.estimator.ModeKeys.PREDICT:
        # Return the predicted values.
        return tf.estimator.EstimatorSpec(
            mode, predictions={
                'predictions': predictions
            })

    # Define the loss function.
    loss_function = tf.keras.losses.MeanSquaredError()

    # Obtain the regularization loss of the model.
    regularization_loss = sum(model.losses)

    # Compute the loss defined previously.
    loss = loss_function(labels, predictions)

    # Compute the total loss.
    total_loss = loss + regularization_loss

    tf.summary.scalar('loss', loss)
    tf.summary.scalar('regularization_loss', regularization_loss)
    tf.summary.scalar('total_loss', total_loss)

    # If we are performing evaluation.
    if mode == tf.estimator.ModeKeys.EVAL:
        # Define the different evaluation metrics
        label_mean = tf.keras.metrics.Mean()
        _ = label_mean.update_state(labels)
        prediction_mean = tf.keras.metrics.Mean()
        _ = prediction_mean.update_state(predictions)
        mae = tf.keras.metrics.MeanAbsoluteError()
        _ = mae.update_state(labels, predictions)
        mre = tf.keras.metrics.MeanRelativeError(normalizer=tf.abs(labels))
        _ = mre.update_state(labels, predictions)

        return tf.estimator.EstimatorSpec(
            mode, loss=loss,
            eval_metric_ops={
                'label/mean': label_mean,
                'prediction/mean': prediction_mean,
                'mae': mae,
                'mre': mre,
                'r-squared': r_squared(labels, predictions)
            }
        )

    # If we are performing training.

    assert mode == tf.estimator.ModeKeys.TRAIN
    # Compute the gradients.
    grads = tf.gradients(total_loss, model.trainable_variables)

    summaries = [tf.summary.histogram(var.op.name, var)
                 for var in model.trainable_variables]
    summaries += [tf.summary.histogram(g.op.name, g)
                  for g in grads if g is not None]

    # Define an exponential decay schedule.
    decayed_lr = tf.keras.optimizers.schedules.ExponentialDecay(float(params['HYPERPARAMETERS']['learning_rate']),
                                                                int(params['HYPERPARAMETERS']
                                                                    ['decay_steps']),
                                                                float(
                                                                    params['HYPERPARAMETERS']['decay_rate']),
                                                                staircase=True)

    # Define an Adam optimizer using the defined exponential decay.
    optimizer = tf.keras.optimizers.Adam(learning_rate=decayed_lr)

    # Manually assign tf.compat.v1.global_step variable to optimizer.iterations
    # to make tf.compat.v1.train.global_step increased correctly.
    optimizer.iterations = tf.compat.v1.train.get_or_create_global_step()

    # Apply the processed gradients using the optimizer.
    train_op = optimizer.apply_gradients(zip(grads, model.trainable_variables))

    # Define the logging hook. It returns the loss, the regularization loss and the
    # total loss every 10 iterations.
    logging_hook = tf.estimator.LoggingTensorHook(
        {"Loss": loss,
         "Regularization loss": regularization_loss,
         "Total loss": total_loss}, every_n_iter=10)

    return tf.estimator.EstimatorSpec(mode,
                                      loss=loss,
                                      train_op=train_op,
                                      training_hooks=[logging_hook]
                                      )
