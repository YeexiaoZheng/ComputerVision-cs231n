from builtins import range
from builtins import object
import numpy as np

from daseCV.layers import *
from daseCV.layer_utils import *


class TwoLayerNet(object):
    """
    采用模块化设计实现具有ReLU和softmax损失函数的两层全连接神经网络。
    假设D是输入维度，H是隐藏层维度，一共有C类标签。
   
    网络架构应该是：affine - relu - affine - softmax.
    
    注意，这个类不实现梯度下降；它将与负责优化的Solver对象进行交互。
    
    模型的可学习参数存储在字典self.params中。键是参数名称，值是numpy数组。
    """

    def __init__(self, input_dim=3*32*32, hidden_dim=100, num_classes=10,
                 weight_scale=1e-3, reg=0.0):
        """
        Initialize a new network.

        Inputs:
        - input_dim: An integer giving the size of the input
        - hidden_dim: An integer giving the size of the hidden layer
        - num_classes: An integer giving the number of classes to classify
        - weight_scale: Scalar giving the standard deviation for random
          initialization of the weights.
        - reg: Scalar giving L2 regularization strength.
        """
        self.params = {}
        self.reg = reg

        ############################################################################
        # TODO: Initialize the weights and biases of the two-layer net. Weights    #
        # should be initialized from a Gaussian centered at 0.0 with               #
        # standard deviation equal to weight_scale, and biases should be           #
        # initialized to zero. All weights and biases should be stored in the      #
        # dictionary self.params, with first layer weights                         #
        # and biases using the keys 'W1' and 'b1' and second layer                 #
        # weights and biases using the keys 'W2' and 'b2'.                         #
        ############################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        
        self.params['W1']=weight_scale*np.random.randn(input_dim,hidden_dim)
        self.params['b1']=np.zeros(hidden_dim)
        self.params['W2']=weight_scale*np.random.randn(hidden_dim,num_classes)
        self.params['b2']=np.zeros(num_classes)

        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################
        
        
    def loss(self, X, y=None):
        """
        对小批量数据计算损失和梯度

        Inputs:
        - X: Array of input data of shape (N, d_1, ..., d_k)
        - y: Array of labels, of shape (N,). y[i] gives the label for X[i].

        Returns:
        If y is None, then run a test-time forward pass of the model and return:
        - scores: Array of shape (N, C) giving classification scores, where
          scores[i, c] is the classification score for X[i] and class c.

        If y is not None, then run a training-time forward and backward pass and
        return a tuple of:
        - loss: Scalar value giving the loss
        - grads: Dictionary with the same keys as self.params, mapping parameter
          names to gradients of the loss with respect to those parameters.
        """
        scores = None
        ############################################################################
        # TODO: Implement the forward pass for the two-layer net, computing the    #
        # class scores for X and storing them in the scores variable.              #
        ############################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        
        layer1_out, layer1_cache = affine_relu_forward(X, self.params['W1'],self.params['b1'])
        layer2_out, layer2_cache = affine_forward(layer1_out,self.params['W2'], self.params['b2'])
        scores = layer2_out

        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        # If y is None then we are in test mode so just return scores
        if y is None:
            return scores

        loss, grads = 0, {}
        ############################################################################
        # TODO: Implement the backward pass for the two-layer net. Store the loss  #
        # in the loss variable and gradients in the grads dictionary. Compute data #
        # loss using softmax, and make sure that grads[k] holds the gradients for  #
        # self.params[k]. Don't forget to add L2 regularization!                   #
        #                                                                          #
        # NOTE: To ensure that your implementation matches ours and you pass the   #
        # automated tests, make sure that your L2 regularization includes a factor #
        # of 0.5 to simplify the expression for the gradient.                      #
        ############################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        
        loss, dscores = softmax_loss(scores, y)
        loss = loss + 0.5 * self.reg * np.sum(self.params['W1']*self.params['W1']) + 0.5 * self.reg * np.sum(self.params['W2'] * self.params['W2'])
        d1_out, dw2, db2 = affine_backward(dscores, layer2_cache)
        grads['W2'] = dw2 + self.reg * self.params['W2']
        grads['b2'] = db2
        dx1, dw1, db1 = affine_relu_backward(d1_out, layer1_cache)
        grads['W1'] = dw1 + self.reg * self.params['W1']
        grads['b1']=db1
        
        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        return loss, grads
    
    
class FullyConnectedNet(object):
    """
    一个任意隐藏层数和神经元数的全连接神经网络，其中 ReLU 激活函数，sofmax 损失函数，同时可选的
    采用 dropout 和 batch normalization(批量归一化)。那么，对于一个L层的神经网络来说，其框架是：
    
    {affine ‐ [batch norm] ‐ relu ‐ [dropout]} x (L ‐ 1) ‐ affine ‐ softmax
    
    其中的[batch norm]和[dropout]是可选非必须的，框架中{...}部分将会重复L‐1次，代表L‐1 个隐藏层。
    
    与我们在上面定义的 TwoLayerNet() 类保持一致，所有待学习的参数都会存在self.params 字典中，
    并且最终会被最优化 Solver() 类训练学习得到。
    """

    def __init__(self, hidden_dims, input_dim=3*32*32, num_classes=10,
                 dropout=1, normalization=None, reg=0.0,
                 weight_scale=1e-2, dtype=np.float32, seed=None):
        """
        Initialize a new FullyConnectedNet.

        Inputs:
        - hidden_dims: A list of integers giving the size of each hidden layer.
        - input_dim: An integer giving the size of the input.
        - num_classes: An integer giving the number of classes to classify.
        - dropout: Scalar between 0 and 1 giving dropout strength. If dropout=1 then
          the network should not use dropout at all.
        - normalization: What type of normalization the network should use. Valid values
          are "batchnorm", "layernorm", or None for no normalization (the default).
        - reg: Scalar giving L2 regularization strength.
        - weight_scale: Scalar giving the standard deviation for random
          initialization of the weights.
        - dtype: A numpy datatype object; all computations will be performed using
          this datatype. float32 is faster but less accurate, so you should use
          float64 for numeric gradient checking.
        - seed: If not None, then pass this random seed to the dropout layers. This
          will make the dropout layers deteriminstic so we can gradient check the
          model.
        """
        self.normalization = normalization
        self.use_dropout = dropout != 1
        self.reg = reg
        self.num_layers = 1 + len(hidden_dims)
        self.dtype = dtype
        self.params = {}

        ############################################################################
        # TODO: Initialize the parameters of the network, storing all values in    #
        # the self.params dictionary. Store weights and biases for the first layer #
        # in W1 and b1; for the second layer use W2 and b2, etc. Weights should be #
        # initialized from a normal distribution centered at 0 with standard       #
        # deviation equal to weight_scale. Biases should be initialized to zero.   #
        #                                                                          #
        # When using batch normalization, store scale and shift parameters for the #
        # first layer in gamma1 and beta1; for the second layer use gamma2 and     #
        # beta2, etc. Scale parameters should be initialized to ones and shift     #
        # parameters should be initialized to zeros.                               #
        ############################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

        self.params["W1"] = np.random.randn(input_dim, hidden_dims[0]) * weight_scale
        self.params["b1"] = np.zeros((hidden_dims[0]))
        
        for i in range(len(hidden_dims) - 1):
            self.params[f"W{i+2}"] = np.random.randn(hidden_dims[i], hidden_dims[i+1]) * weight_scale
            self.params[f"b{i+2}"] = np.zeros((hidden_dims[i+1]))
            if not self.normalization == None:
                self.params[f"gamma{i + 1}"] = np.ones(hidden_dims[i])
                self.params[f"beta{i + 1}"] = np.zeros((hidden_dims[i]))
        
        self.params[f"W{self.num_layers}"] = np.random.randn(hidden_dims[-1], num_classes) * weight_scale
        self.params[f"b{self.num_layers}"] = np.zeros((num_classes))
        
        if not self.normalization == None:
            self.params[f"gamma{len(hidden_dims)}"]  = np.ones(hidden_dims[-1])
            self.params[f"beta{len(hidden_dims)}"]  = np.zeros((hidden_dims[-1]))

        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        # When using dropout we need to pass a dropout_param dictionary to each
        # dropout layer so that the layer knows the dropout probability and the mode
        # (train / test). You can pass the same dropout_param to each dropout layer.
        self.dropout_param = {}
        if self.use_dropout:
            self.dropout_param = {'mode': 'train', 'p': dropout}
            if seed is not None:
                self.dropout_param['seed'] = seed

        # With batch normalization we need to keep track of running means and
        # variances, so we need to pass a special bn_param object to each batch
        # normalization layer. You should pass self.bn_params[0] to the forward pass
        # of the first batch normalization layer, self.bn_params[1] to the forward
        # pass of the second batch normalization layer, etc.
        self.bn_params = []
        if self.normalization=='batchnorm':
            self.bn_params = [{'mode': 'train'} for i in range(self.num_layers - 1)]
        if self.normalization=='layernorm':
            self.bn_params = [{} for i in range(self.num_layers - 1)]

        # Cast all parameters to the correct datatype
        for k, v in self.params.items():
            self.params[k] = v.astype(dtype)


    def loss(self, X, y=None):
        """
        Compute loss and gradient for the fully-connected net.

        Input / output: Same as TwoLayerNet above.
        """
        X = X.astype(self.dtype)
        mode = 'test' if y is None else 'train'

        # Set train/test mode for batchnorm params and dropout param since they
        # behave differently during training and testing.
        if self.use_dropout:
            self.dropout_param['mode'] = mode
        if self.normalization=='batchnorm':
            for bn_param in self.bn_params:
                bn_param['mode'] = mode
        scores = None
        ############################################################################
        # TODO: Implement the forward pass for the fully-connected net, computing  #
        # the class scores for X and storing them in the scores variable.          #
        #                                                                          #
        # When using dropout, you'll need to pass self.dropout_param to each       #
        # dropout forward pass.                                                    #
        #                                                                          #
        # When using batch normalization, you'll need to pass self.bn_params[0] to #
        # the forward pass for the first batch normalization layer, pass           #
        # self.bn_params[1] to the forward pass for the second batch normalization #
        # layer, etc.                                                              #
        ############################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

        cache_list = []
        dropout_caches = []
        next_input = np.copy(X)
        
        for i in range(self.num_layers - 1):
            if self.normalization=='batchnorm':
                next_input, cache = affine_bnorm_relu_forward(next_input, \
                                                              self.params[f"W{i+1}"], \
                                                              self.params[f"b{i+1}"], \
                                                              self.params[f"gamma{i+1}"],\
                                                              self.params[f"beta{i+1}"],\
                                                              self.bn_params[i]
                                                             )
            elif self.normalization=='layernorm':
                next_input, cache = affine_lnorm_relu_forward(next_input, \
                                                              self.params[f"W{i+1}"], \
                                                              self.params[f"b{i+1}"], \
                                                              self.params[f"gamma{i+1}"],\
                                                              self.params[f"beta{i+1}"],\
                                                              self.bn_params[i]
                                                             )

            else:
                next_input, cache = affine_relu_forward(next_input, \
                                                        self.params[f"W{i+1}"], \
                                                        self.params[f"b{i+1}"])
            cache_list.append(cache)
            
            if self.use_dropout:
                next_input, dropout_cache = dropout_forward(next_input, self.dropout_param)
                dropout_caches.append(dropout_cache)

        scores, out_cache = affine_forward(next_input, self.params[f"W{self.num_layers}"],\
                                           self.params[f"b{self.num_layers}"])
        cache_list.append(out_cache)

        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        # If test mode return early
        if mode == 'test':
            return scores

        loss, grads = 0.0, {}
        ############################################################################
        # TODO: Implement the backward pass for the fully-connected net. Store the #
        # loss in the loss variable and gradients in the grads dictionary. Compute #
        # data loss using softmax, and make sure that grads[k] holds the gradients #
        # for self.params[k]. Don't forget to add L2 regularization!               #
        #                                                                          #
        # When using batch/layer normalization, you don't need to regularize the scale   #
        # and shift parameters.                                                    #
        #                                                                          #
        # NOTE: To ensure that your implementation matches ours and you pass the   #
        # automated tests, make sure that your L2 regularization includes a factor #
        # of 0.5 to simplify the expression for the gradient.                      #
        ############################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

        loss, dout = softmax_loss(scores, y)
        sq_sum = 0
        
        for i in range(self.num_layers):
            sq_sum += np.sum(self.params[f"W{i + 1}"] ** 2)
            
        loss += 0.5 * self.reg * sq_sum
        
        next_grad, grads[f"W{self.num_layers}"], grads[f"b{self.num_layers}"] = affine_backward(dout, cache_list[-1])
        grads[f"W{self.num_layers}"] += self.reg * self.params[f"W{self.num_layers}"]
        
        for i in range(self.num_layers - 1).__reversed__():

            if self.use_dropout:
                next_grad = dropout_backward(next_grad, dropout_caches[i])

            if self.normalization == 'batchnorm':
                next_grad, grads[f"W{i+1}"], grads[f"b{i+1}"], grads[f"gamma{i+1}"], grads[f"beta{i+1}"] = affine_bnorm_relu_backward(next_grad, cache_list[i])

            elif self.normalization == 'layernorm':
                next_grad, grads[f"W{i+1}"], grads[f"b{i+1}"], grads[f"gamma{i+1}"], grads[f"beta{i+1}"] = affine_lnorm_relu_backward(next_grad, cache_list[i])

            else:
                next_grad, grads[f"W{i+1}"], grads[f"b{i+1}"] = affine_relu_backward(next_grad, cache_list[i])
            grads[f"W{i+1}"] += self.reg * self.params[f"W{i+1}"]

        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        return loss, grads
