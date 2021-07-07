import tensorflow as tf
from tensorflow.keras.layers import BatchNormalization, Activation, Dense, Add
import datetime


class DenseNet(tf.keras.Model):

    def __init__(self):
        super(DenseNet, self).__init__()
        
        # =====================================================
        # =                  ----------------------------------
        # = H-PARAM & CONFIG ==================================
        # =                  ----------------------------------
        # =====================================================
        
        NUM_OF_CLASS        = 2
        USE_BIAS            = True
        EPSILON             = 1e-3
        
        KERNEL_INITIALIZER  = None#'he_uniform'
        KERNEL_REGULARIZER  = tf.keras.regularizers.l2(0.001)
        BIAS_REGULARIZER    = tf.keras.regularizers.l2(0.001)
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
        
        # ======================================================
        # =         --------------------------------------------
        # = dense_2 ============================================
        # =         --------------------------------------------
        # ======================================================
        self.dense_2_1 = Dense(
            units              = 8,
            activation         = None,
            use_bias           = USE_BIAS,
            kernel_initializer = KERNEL_INITIALIZER,
            kernel_regularizer = KERNEL_REGULARIZER,
            bias_regularizer   = BIAS_REGULARIZER
        )
        
        self.bn_2_1 = tf.keras.layers.BatchNormalization(
            epsilon           = EPSILON,
            beta_regularizer  = BETA_REGULARIZER,
            gamma_regularizer = GAMMA_REGULARIZER, 
        )
        
        self.relu_2_1 = tf.keras.layers.ReLU()
        
        # ======================================================
        # =         --------------------------------------------
        # = dense_3 ============================================
        # =         --------------------------------------------
        # ======================================================
        self.dense_3_1 = Dense(
            units              = 8,
            activation         = None,
            use_bias           = USE_BIAS,
            kernel_initializer = KERNEL_INITIALIZER,
            kernel_regularizer = KERNEL_REGULARIZER,
            bias_regularizer   = BIAS_REGULARIZER
        )
        
        self.bn_3_1 = tf.keras.layers.BatchNormalization(
            epsilon           = EPSILON,
            beta_regularizer  = BETA_REGULARIZER,
            gamma_regularizer = GAMMA_REGULARIZER, 
        )
        
        self.relu_3_1 = tf.keras.layers.ReLU()
        
        # ======================================================
        # =         --------------------------------------------
        # = dense_4 ============================================
        # =         --------------------------------------------
        # ======================================================
        self.dense_4_1 = Dense(
            units              = 8,
            activation         = None,
            use_bias           = USE_BIAS,
            kernel_initializer = KERNEL_INITIALIZER,
            kernel_regularizer = KERNEL_REGULARIZER,
            bias_regularizer   = BIAS_REGULARIZER
        )
        
        self.bn_4_1 = tf.keras.layers.BatchNormalization(
            epsilon           = EPSILON,
            beta_regularizer  = BETA_REGULARIZER,
            gamma_regularizer = GAMMA_REGULARIZER, 
        )
        
        self.relu_4_1 = tf.keras.layers.ReLU()
        
        # ======================================================
        # =         --------------------------------------------
        # = dense_5 ============================================
        # =         --------------------------------------------
        # ======================================================
        self.dense_5_1 = Dense(
            units              = 8,
            activation         = None,
            use_bias           = USE_BIAS,
            kernel_initializer = KERNEL_INITIALIZER,
            kernel_regularizer = KERNEL_REGULARIZER,
            bias_regularizer   = BIAS_REGULARIZER
        )
        
        self.bn_5_1 = tf.keras.layers.BatchNormalization(
            epsilon           = EPSILON,
            beta_regularizer  = BETA_REGULARIZER,
            gamma_regularizer = GAMMA_REGULARIZER, 
        )
        
        self.relu_5_1 = tf.keras.layers.ReLU()
        
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
        
        '''
        # =====================================================
        # =        --------------------------------------------
        # = dense_2 ============================================
        # =        --------------------------------------------
        # =====================================================
        
        x = self.dense_2_1(inputs)
        x = self.bn_2_1(x, training)
        x = self.relu_2_1(x)
        
        # =====================================================
        # =        --------------------------------------------
        # = dense_3 ============================================
        # =        --------------------------------------------
        # =====================================================
        
        x = self.dense_3_1(inputs)
        x = self.bn_3_1(x, training)
        x = self.relu_3_1(x)
        
        # =====================================================
        # =        --------------------------------------------
        # = dense_4 ============================================
        # =        --------------------------------------------
        # =====================================================
        
        x = self.dense_4_1(inputs)
        x = self.bn_4_1(x, training)
        x = self.relu_4_1(x)
        
        # =====================================================
        # =        --------------------------------------------
        # = dense_5 ============================================
        # =        --------------------------------------------
        # =====================================================
        
        x = self.dense_5_1(inputs)
        x = self.bn_5_1(x, training)
        x = self.relu_5_1(x)
        '''
        # =====================================================
        # =       ---------------------------------------------
        # = DENSE =============================================
        # =       ---------------------------------------------
        # =====================================================
        
        
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