
from Data import Dataset
from GAN.Model import DRAGAN
import tensorflow as tf
from tensorflow import keras
import tqdm
import functools
import pylib as py
import imlib as im

class AnimeDragan:

    def __init__(self):

        self.model = DRAGAN()

        self.dataDirectory = "./archive/animefaces256cleaner"
        self.batchSize = 32
        self.datasetImageHeight = 128
        self.datasetImageWidth = 128

    def trainGAN(self):

        self.dataset, shape, self.datasetLength = Dataset(self.dataDirectory, self.batchSize, self.datasetImageHeight, self.datasetImageWidth)
        n_G_upsamplings = n_D_downsamplings = 4

        self.G = self.model.ConvGenerator(input_shape=(1, 1, 128), output_channels=shape[-1], n_upsamplings=n_G_upsamplings)
        self.D = self.model.ConvDiscriminator(input_shape=shape, n_downsamplings=n_D_downsamplings)
        
        self.G_optimizer = keras.optimizers.Adam(learning_rate=.0002, beta_1=0.5)
        self.D_optimizer = keras.optimizers.Adam(learning_rate=.0002, beta_1=0.5)

    
         
   
    def d_loss_fn(self,r_logit, f_logit):
        r_loss = - tf.reduce_mean(r_logit)
        f_loss = tf.reduce_mean(f_logit)
        return r_loss, f_loss

    def g_loss_fn(self, f_logit):
        f_loss = - tf.reduce_mean(f_logit)
        return f_loss

    @tf.function
    def train_G(self):
        with tf.GradientTape() as t:
            z = tf.random.normal(shape=(self.batchsize, 1, 1, 128))
            x_fake = self.G(z, training=True)
            x_fake_d_logit = self.D(x_fake, training=True)
            G_loss = self.g_loss_fn(x_fake_d_logit)

        G_grad = t.gradient(G_loss, self.G.trainable_variables)
        self.G_optimizer.apply_gradients(zip(G_grad, self.G.trainable_variables))

        return {'g_loss': G_loss}


    @tf.function
    def train_D(self, x_real):
        with tf.GradientTape() as t:
            z = tf.random.normal(shape=(self.batchsize, 1, 1, 128))
            x_fake = self.G(z, training=True)

            x_real_d_logit = self.D(x_real, training=True)
            x_fake_d_logit = self.D(x_fake, training=True)

            x_real_d_loss, x_fake_d_loss = self.d_loss_fn(x_real_d_logit, x_fake_d_logit)
            gp = self.gradient_penalty(functools.partial(self.D, training=True), x_real, x_fake, mode='dragan')

            D_loss = (x_real_d_loss + x_fake_d_loss) + gp * 10

        D_grad = t.gradient(D_loss, self.D.trainable_variables)
        self.D_optimizer.apply_gradients(zip(D_grad, self.D.trainable_variables))

        return {'d_loss': x_real_d_loss + x_fake_d_loss, 'gp': gp}
   
    @tf.function
    def sample(self,z):
        return self.G(z, training=False)

    def gradient_penalty(f, real, fake, mode):
        def _gradient_penalty(f, real, fake=None):
            def _interpolate(a, b=None):
                if b is None:   # interpolation in DRAGAN
                    beta = tf.random.uniform(shape=tf.shape(a), minval=0., maxval=1.)
                    b = a + 0.5 * tf.math.reduce_std(a) * beta
                shape = [tf.shape(a)[0]] + [1] * (a.shape.ndims - 1)
                alpha = tf.random.uniform(shape=shape, minval=0., maxval=1.)
                inter = a + alpha * (b - a)
                inter.set_shape(a.shape)
                return inter

            x = _interpolate(real, fake)
            with tf.GradientTape() as t:
                t.watch(x)
                pred = f(x)
            grad = t.gradient(pred, x)
            norm = tf.norm(tf.reshape(grad, [tf.shape(grad)[0], -1]), axis=1)
            gp = tf.reduce_mean((norm - 1.)**2)

            return gp

        if mode == 'none':
            gp = tf.constant(0, dtype=real.dtype)
        elif mode == 'dragan':
            gp = _gradient_penalty(f, real)
        elif mode == 'wgan-gp':
            gp = _gradient_penalty(f, real, fake)

        return gp




if __name__ == "__main___":
    ep_cnt = tf.Variable(initial_value=0, trainable=False, dtype=tf.int64)
    output = 'Saved Model'

    sample_dir = py.join(output, 'samples_training')
    py.mkdir(sample_dir)
    train_summary_writer = tf.summary.create_file_writer(py.join(output, 'summaries', 'train'))

    modelTrainer = AnimeDragan()
    modelTrainer.trainGan()

    z= tf.random.normal((100, 1, 1, 128))
    with train_summary_writer.as_default():
        for ep in tqdm.trange(500, desc='Epoch loop'):
            if ep < ep_cnt:
                continue

            ep_cnt.assign_add(1)

        # train for an epoch
        for x_real in tqdm.tqdm(modelTrainer.dataset, desc='Inner Epoch Loop', total=modelTrainer.datasetLength):
            D_loss_dict = modelTrainer.train_D(x_real)

            if modelTrainer.D_optimizer.iterations.numpy() % 1 == 0:
                G_loss_dict = modelTrainer.train_G()

            # sample
            if modelTrainer.G_optimizer.iterations.numpy() % 100 == 0:
                x_fake = modelTrainer.sample(z)
                img = im.immerge(x_fake, n_rows=10).squeeze()
                im.imwrite(img, py.join(sample_dir, 'iter-%09d.jpg' % modelTrainer.G_optimizer.iterations.numpy()))

