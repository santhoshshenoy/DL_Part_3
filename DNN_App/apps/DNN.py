
import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, BatchNormalization, Activation, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.initializers import RandomNormal,Constant


class DNN_Model() :

    def __init__(self,type='regression'):
        self.type = type

    def create_model(self,in_shape,outs,**kwargs):
        self.in_shape = in_shape
        
        if kwargs != {}:
            self.num_layers = kwargs['num_layers'] if 'num_layers' in kwargs else 1
            self.batch_size = kwargs['batch_size'] if 'batch_size' in kwargs else 10
            self.activation = kwargs['activation'] if 'activation' in kwargs else 'relu'
            self.kernel_init = kwargs['kernel_init'] if 'kernel_init' in kwargs else 'uniform'
            # self.optimizer = kwargs['opts'] if 'opts' in kwargs else tf.keras.optimizers.Adam(learning_rate=0.001)
            self.neurons_per_layer = kwargs['neurons_per_layer'] if 'neurons_per_layer' in kwargs else [64]*self.num_layers
            self.drop_rate_per_layer = kwargs['drop_rate_per_layer'] if 'drop_rate_per_layer' in kwargs else [0.2]*self.num_layers
            self.n_epochs = kwargs['n_epochs'] if 'n_epochs' in kwargs else 50
            self.arch = kwargs['arch'] if 'arch' in kwargs else 'B A D'
            self.momentum = kwargs['momentum'] if 'meomentum' in kwargs else 0.99
            self.epsilon = kwargs['epsilon'] if 'epsilon' in kwargs else 0.005
            self.beta_init_std = kwargs['beta_init_std'] if 'beta_init_std' in kwargs else 0.05
            self.gamma_init = kwargs['gamma_init'] if 'gamma_init' in kwargs else 0.9
            self.center = kwargs['center'] if 'center' in kwargs else True
            self.scale = kwargs['scale'] if 'scale' in kwargs else False

            if 'opts' in kwargs:
            # check if the kwargs is a class object or a dictionary
                if hasattr(kwargs['opts'],'_name'):
                    self.optimizer = kwargs['opts']
                else:
                    opt_conf = kwargs['opts']
                    use_opt = opt_conf['optimizer']
                    opt_kwargs = {k:opt_conf[k] for k in opt_conf.keys() if k != 'optimizer'}
                    try:
                        self.optimizer = getattr(tf.optimizers, use_opt)(**opt_kwargs)
                    except:
                        self.optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)

            else:
                self.optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
            
        else:
            self.num_layers = 1
            self.batch_size = 10
            self.activation = 'relu'
            self.kernel_init = 'normal'
            self.optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
            self.neurons_per_layer = [32] * self.num_layers
            self.drop_rate_per_layer = [0.2] * self.num_layers
            self.n_epochs = 50
            self.arch = 'B A D'
            self.momentum = 0.99
            self.epsilon = 0.005
            self.beta_init_std = 0.05
            self.gamma_init = 0.9
            self.center = True
            self.scale = False

            
        if self.type == 'regression':
            self.op_activation = 'linear'
            self.outs = 1
        elif self.type == 'classification':
            self.op_activation = 'softmax'
            self.outs = outs
        
        inputs = Input(shape=self.in_shape)
        
        for layer in range(self.num_layers):
            if layer == 0 :
                m = Dense(self.neurons_per_layer[layer],kernel_initializer = self.kernel_init)(inputs)
            else:
                m = Dense(self.neurons_per_layer[layer],kernel_initializer = self.kernel_init)(m)

            for keys in self.arch.split(' '):
                if keys == 'A':
                    m = Activation(self.activation)(m)
                elif keys == 'B':
                    m = BatchNormalization(momentum = self.momentum,
                                            epsilon = self.epsilon,
                                            beta_initializer = RandomNormal(mean=0.0, stddev=self.beta_init_std),
                                            gamma_initializer = Constant(value=self.gamma_init),
                                            center = self.center,
                                            scale = self.scale)(m)
                elif keys == 'D':
                    m = Dropout(self.drop_rate_per_layer[layer])(m)

        outputs = Dense(self.outs, activation=self.op_activation,kernel_initializer = self.kernel_init)(m)

        model = Model(inputs=inputs, outputs=outputs)
        
        return(model)
    
    
    def model_summary(self,model):
        return model.summary()
    

    def train_model(self,model,X_train,y_train,loss,metric,callbacks=None,validation_split=0.2,validation_data=None,ret_result=False,verbose=1):
        self.model = model
        self.loss = loss
        self.metric = metric
        
        self.model.compile(loss=self.loss,
                           optimizer=self.optimizer,
                           metrics=self.metric)
        
        if callbacks == None:
                callbacks = []
        else :
                callbacks = callbacks

        if validation_data == None:
            self.history = self.model.fit(X_train,
                                          y_train,
                                          epochs=self.n_epochs,
                                          validation_split=validation_split,
                                          callbacks=callbacks,
                                          verbose=0)

        else:
            self.history = self.model.fit(X_train,
                                          y_train,
                                          epochs=self.n_epochs,
                                          validation_data=validation_data,
                                          callbacks=callbacks,
                                          verbose=0)
        if ret_result == True:
            return self.model.evaluate(X_train,y_train,batch_size=self.batch_size,verbose=0)
        else:
            return None
            
    
    def test_model(self,X_test,y_test,callbacks=None):
        self.test_results = self.model.evaluate(X_test,y_test,batch_size=self.batch_size,callbacks=callbacks,verbose=0)
        return self.test_results
        
        
        
        