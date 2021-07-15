import tensorflow as tf
from tensorflow.keras.layers import BatchNormalization, Activation, Dense, Add
import datetime


class Classification(tf.keras.Model):

    def __init__(self):
        super(Classification, self).__init__()
        
        # =====================================================
        # =                  ----------------------------------
        # = H-PARAM & CONFIG ==================================
        # =                  ----------------------------------
        # =====================================================
        
        NUM_OF_CLASS        = 2
        USE_BIAS            = True
        EPSILON             = 1e-3
        
        KERNEL_INITIALIZER  = 'he_uniform'
        KERNEL_REGULARIZER  = tf.keras.regularizers.l2(0.01)
        BIAS_REGULARIZER    = tf.keras.regularizers.l2(0.0001)
        BETA_REGULARIZER    = tf.keras.regularizers.l2(0.0)
        GAMMA_REGULARIZER   = tf.keras.regularizers.l2(0.0)
        
        # ======================================================
        # =         --------------------------------------------
        # = dense_1 ============================================
        # =         --------------------------------------------
        # ======================================================
        self.dense_1_1 = Dense(
            units              = 8,
            activation         = None,
            use_bias           = USE_BIAS,
            kernel_initializer = KERNEL_INITIALIZER,
            kernel_regularizer = KERNEL_REGULARIZER,
            bias_regularizer   = BIAS_REGULARIZER
        )
        
        self.bn_1_1 = tf.keras.layers.BatchNormalization(
            epsilon           = EPSILON,
            beta_regularizer  = BETA_REGULARIZER,
            gamma_regularizer = GAMMA_REGULARIZER, 
        )
        
        self.relu_1_1 = tf.keras.layers.ReLU()
        
        # =====================================================
        # =        --------------------------------------------
        # = dense_2 ============================================
        # =        --------------------------------------------
        # =====================================================
        self.dense_2_1, self.bn_2_1 = [], []
        self.dense_2_2, self.bn_2_2 = [], []
        self.relu_2_1, self.relu_2_2 =[], []
        self.shortcut_connect_2 = []
        for _ in range(3):
            self.dense_2_1.append(
                Dense(
                    units              = 8,
                    activation         = None,
                    use_bias           = USE_BIAS,
                    kernel_initializer = KERNEL_INITIALIZER,
                    kernel_regularizer = KERNEL_REGULARIZER,
                    bias_regularizer   = BIAS_REGULARIZER
                )
            )
            
            self.bn_2_1.append(
                tf.keras.layers.BatchNormalization(
                    epsilon=EPSILON,
                    beta_regularizer=BETA_REGULARIZER,
                    gamma_regularizer=GAMMA_REGULARIZER, 
                )
            )
            self.relu_2_1.append(
                tf.keras.layers.ReLU()
            )
            
            
            self.dense_2_2.append(
                Dense(
                    units              = 8,
                    activation         = None,
                    use_bias           = USE_BIAS,
                    kernel_initializer = KERNEL_INITIALIZER,
                    kernel_regularizer = KERNEL_REGULARIZER,
                    bias_regularizer   = BIAS_REGULARIZER
                )
            )
            
            self.bn_2_2.append(
                tf.keras.layers.BatchNormalization(
                    epsilon=EPSILON,
                    beta_regularizer=BETA_REGULARIZER,
                    gamma_regularizer=GAMMA_REGULARIZER, 
                )
            )
        
            self.relu_2_2.append(
                tf.keras.layers.ReLU()
            )
            
            self.shortcut_connect_2.append(
                tf.keras.layers.Add()
            )
            
        # ======================================================
        # =         --------------------------------------------
        # = dense_3 ============================================
        # =         --------------------------------------------
        # ======================================================
        self.dense_3_shortcut = Dense(
                    units              = 16,
                    activation         = None,
                    use_bias           = USE_BIAS,
                    kernel_initializer = KERNEL_INITIALIZER,
                    kernel_regularizer = KERNEL_REGULARIZER,
                    bias_regularizer   = BIAS_REGULARIZER
                )
        self.relu_3_shortcut = tf.keras.layers.ReLU()
        self.dense_3_1, self.bn_3_1 = [], []
        self.dense_3_2, self.bn_3_2 = [], []
        self.relu_3_1, self.relu_3_2 =[], []
        self.shortcut_connect_3 = []
        for i in range(3):
            self.dense_3_1.append(
                Dense(
                    units              = 16,
                    activation         = None,
                    use_bias           = USE_BIAS,
                    kernel_initializer = KERNEL_INITIALIZER,
                    kernel_regularizer = KERNEL_REGULARIZER,
                    bias_regularizer   = BIAS_REGULARIZER
                )
            )
            self.relu_3_1.append(
                tf.keras.layers.ReLU()
            )
            self.bn_3_1.append(
                tf.keras.layers.BatchNormalization(
                    epsilon=EPSILON,
                    beta_regularizer=BETA_REGULARIZER,
                    gamma_regularizer=GAMMA_REGULARIZER, 
                )
            )
            
            
            self.dense_3_2.append(
                Dense(
                    units              = 16,
                    activation         = None,
                    use_bias           = USE_BIAS,
                    kernel_initializer = KERNEL_INITIALIZER,
                    kernel_regularizer = KERNEL_REGULARIZER,
                    bias_regularizer   = BIAS_REGULARIZER
                )
            )
            self.bn_3_2.append(
                tf.keras.layers.BatchNormalization(
                    epsilon=EPSILON,
                    beta_regularizer=BETA_REGULARIZER,
                    gamma_regularizer=GAMMA_REGULARIZER, 
                )
            )
            self.relu_3_2.append(
                tf.keras.layers.ReLU()
            )
            
            self.shortcut_connect_3.append(
                tf.keras.layers.Add()
            )
            
        # ======================================================
        # =         --------------------------------------------
        # = dense_4 ============================================
        # =         --------------------------------------------
        # ======================================================
        self.dense_4_shortcut = Dense(
                    units              = 8,
                    activation         = None,
                    use_bias           = USE_BIAS,
                    kernel_initializer = KERNEL_INITIALIZER,
                    kernel_regularizer = KERNEL_REGULARIZER,
                    bias_regularizer   = BIAS_REGULARIZER
                )
        self.relu_4_shortcut = tf.keras.layers.ReLU()
        
        
        self.dense_4_1, self.bn_4_1 = [], []
        self.dense_4_2, self.bn_4_2 = [], []
        self.relu_4_1, self.relu_4_2 =[], []
        self.shortcut_connect_4 = []
        for i in range(3):
            self.dense_4_1.append(
                Dense(
                    units              = 8,
                    activation         = None,
                    use_bias           = USE_BIAS,
                    kernel_initializer = KERNEL_INITIALIZER,
                    kernel_regularizer = KERNEL_REGULARIZER,
                    bias_regularizer   = BIAS_REGULARIZER
                )
            )
            self.bn_4_1.append(
                tf.keras.layers.BatchNormalization(
                    epsilon=EPSILON,
                    beta_regularizer=BETA_REGULARIZER,
                    gamma_regularizer=GAMMA_REGULARIZER, 
                )
            )
            self.relu_4_1.append(
                tf.keras.layers.ReLU()
            )
            
            
            self.dense_4_2.append(
                Dense(
                    units              = 8,
                    activation         = None,
                    use_bias           = USE_BIAS,
                    kernel_initializer = KERNEL_INITIALIZER,
                    kernel_regularizer = KERNEL_REGULARIZER,
                    bias_regularizer   = BIAS_REGULARIZER
                )
            )
            self.bn_4_2.append(
                tf.keras.layers.BatchNormalization(
                    epsilon=EPSILON,
                    beta_regularizer=BETA_REGULARIZER,
                    gamma_regularizer=GAMMA_REGULARIZER, 
                )
            )
            self.relu_4_2.append(
                tf.keras.layers.ReLU()
            )
            
            self.shortcut_connect_4.append(
                tf.keras.layers.Add()
            )
            
        # =====================================================
        # =       ---------------------------------------------
        # = DENSE =============================================
        # =       ---------------------------------------------
        # =====================================================
        
        self.dense = Dense(
            units              = NUM_OF_CLASS,
            activation         = 'softmax',
            use_bias           = USE_BIAS,
            kernel_initializer = KERNEL_INITIALIZER,
            kernel_regularizer = KERNEL_REGULARIZER,
            bias_regularizer   = BIAS_REGULARIZER
        )
        
        
        
        #self.flatten = tf.keras.layers.Flatten()
        
        
        # =====================================================
        # =                ------------------------------------
        # = LAYER SET DONE ====================================
        # =                ------------------------------------
        # =====================================================
        
    
    @tf.function
    def call(self, inputs, training=False):
        
        x = inputs
        
        
        # =====================================================
        # =        --------------------------------------------
        # = dense_1 ============================================
        # =        --------------------------------------------
        # =====================================================
        
        x = self.dense_1_1(inputs)
        x = self.bn_1_1(x, training)
        x = self.relu_1_1(x)
        
        
        
        # =====================================================
        # =        --------------------------------------------
        # = dense_2 ============================================
        # =        --------------------------------------------
        # =====================================================
        
        shortcut = x
        for i in range(3):
            x = self.dense_2_1[i](x)
            x = self.bn_2_1[i](x, training)
            x = self.relu_2_1[i](x)
            
            x = self.dense_2_2[i](x)
            x = self.bn_2_2[i](x, training)
            x = self.relu_2_2[i](x)
            
            x = self.shortcut_connect_2[i]([x, shortcut])
            
            shortcut = x
        
        
        # =====================================================
        # =        --------------------------------------------
        # = dense_3 ============================================
        # =        --------------------------------------------
        # =====================================================
                
        shortcut = self.dense_3_shortcut(shortcut)
        shortcut = self.relu_3_shortcut(shortcut)
        for i in range(3):
            x = self.dense_3_1[i](x)
            x = self.bn_3_1[i](x, training)
            x = self.relu_3_1[i](x)
            
            x = self.dense_3_2[i](x)
            x = self.bn_3_2[i](x, training)
            x = self.relu_3_2[i](x)
            
            x = self.shortcut_connect_3[i]([x, shortcut])
            
            shortcut = x
        
        
        # =====================================================
        # =        --------------------------------------------
        # = dense_4 ============================================
        # =        --------------------------------------------
        # =====================================================
        
        shortcut = self.dense_4_shortcut(shortcut)
        shortcut = self.relu_4_shortcut(shortcut)
        for i in range(3):
            x = self.dense_4_1[i](x)
            x = self.bn_4_1[i](x, training)
            x = self.relu_4_1[i](x)
            
            x = self.dense_4_2[i](x)
            x = self.bn_4_2[i](x, training)
            x = self.relu_4_2[i](x)
            
            x = self.shortcut_connect_4[i]([x, shortcut])
            
            shortcut = x
        
        # =====================================================
        # =       ---------------------------------------------
        # = DENSE =============================================
        # =       ---------------------------------------------
        # =====================================================
        
        
        #x = self.avg_pooling(x)
        x = self.dense(x)
        
        return x
    
    
    def trace_graph(self,input_shape):
        
        current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        graph_log_dir = 'logs/Graph/' + current_time + '/graph'
        graph_writer = tf.summary.create_file_writer(graph_log_dir)

        tf.summary.trace_on(graph=True)
        self.call(tf.zeros(input_shape))
        with graph_writer.as_default():
            tf.summary.trace_export(
                name="model_trace",
                step=0)