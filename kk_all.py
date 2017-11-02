from keratin.metrics import dice, dice_loss, hausdorff, dice_weighted, dice_weighted_loss
import matplotlib
matplotlib.use("agg")
import numpy as np
from keratin.networks import unet, unet_with_dropout
from keras.optimizers import Adam
from skimage.transform import resize
import keras
import matplotlib.pyplot as plt
import numpy as np
from glob import glob
from os.path import join, exists
from os import makedirs
from skimage.transform import AffineTransform, matrix_transform, warp
from skimage.morphology import dilation, erosion
from skimage.morphology import disk
import pandas as pd
from nipype.utils.filemanip import load_json, save_json
import sys


def get_model():
    model = unet(256,256,n_channels=2)
    model.compile(optimizer=Adam(lr=10e-6),
              loss=dice_weighted_loss,
              metrics=[dice_weighted, dice])
    return model



def make256(images, hints = None):

    if not hints:
        bigM = np.zeros((len(images), 256, 256, 1))
    else:
        bigM = np.zeros((len(images), 256, 256, 2))

    for i, im in enumerate(images):

        data = plt.imread(im)
        if len(data.shape) == 3:
            do_mean = True
            data = (data[:,:,0]/255).astype(np.float32)
            if hints:
                hint = plt.imread(hints[i]).astype(np.float32)

        else:
            do_mean = False
            #print("mean_data", np.mean(data))
            data = (data/np.max(data)).astype(np.float32)
            #print("mean data", np.mean(data))

        if data.shape[0] > 256:
            data = data[:256, :]
        if data.shape[1] > 256:
            data = data[:, :256]

        data_pad = np.pad(data, (((256-data.shape[0])//2, ((256-data.shape[0]) + (data.shape[0]%2 >0))//2),
                                 ((256-data.shape[1])//2, ((256-data.shape[1]) + (data.shape[1]%2 >0))//2)),
                          "constant", constant_values = (0,0))

        if hints:

            if hint.shape[0] > 256:
                hint = hint[:256, :]
            if hint.shape[1] > 256:
                hint = hint[:, :256]

            hint_pad = np.pad(hint, (((256-hint.shape[0])//2, ((256-hint.shape[0]) + (hint.shape[0]%2 >0))//2),
                         ((256-hint.shape[1])//2, ((256-hint.shape[1]) + (hint.shape[1]%2 >0))//2)),
                  "constant", constant_values = (0,0))


        if do_mean:
            bigM[i,:,:,0] = (data_pad - np.mean(data_pad)) / np.std(data_pad)
            if hints:
                bigM[i,:,:,1] = (hint_pad - np.mean(hint_pad)) / np.std(hint_pad)

            #bigM_mean = np.mean(bigM)
            #bigM_std = np.std(bigM)
            #bigM = (bigM - bigM_mean)/bigM_std
        else:
            bigM[i,:,:,0] = data_pad

    return bigM


# In[4]:

def get_data(images, hints, masks):
    bigM_base = make256(images, hints)
    print("base shape", bigM_base.shape)
    bigM_mask = make256(masks)
    print("mask shape", bigM_mask.shape)
    return bigM_base, bigM_mask


# In[5]:

def get_split_indices(subjects, subjects_all):
    idx = list(range(len(set(subjects))))
    np.random.shuffle(idx)
    train_subs = idx[:int(0.8*subjects.shape[0])]
    test_subs = idx[int(0.8*subjects.shape[0]):int(0.9*subjects.shape[0])]
    val_subs = idx[int(0.9*subjects.shape[0]):]
    train = [i for i, val in enumerate(subjects_all) if val in subjects[train_subs]]
    np.random.shuffle(train)
    test = [i for i, val in enumerate(subjects_all) if val in subjects[test_subs]]
    np.random.shuffle(test)
    val = [i for i, val in enumerate(subjects_all) if val in subjects[val_subs]]
    np.random.shuffle(val)
    return train, test, val


# In[6]:

def get_random_affine():
    rotation = np.random.rand()*np.pi/45/2 * (np.random.binomial(1,0.5) * 2 - 1) # +- 4 degrees
    shear = np.random.rand()*np.pi/45/2 * (np.random.binomial(1,0.5) * 2 - 1) # +- 4 degrees
    translation = [t * (np.random.binomial(1,0.5) * 2 - 1) for t in np.random.rand(2) * 10]
    scale = [1 + (t * (np.random.binomial(1,0.5) * 2 - 1)) for t in (np.random.rand(2) / 10)]
    #print("r", rotation, "s", shear, "t", translation, "sc", scale)
    return AffineTransform(scale=scale, rotation=rotation, shear=shear, translation=translation)


# In[7]:

def wiggle_image(data, truth):
    xfm = get_random_affine()
    return warp(data, xfm), warp(truth, xfm)


# In[8]:

def augment_data(x_arr, y_arr):
    X = np.zeros(x_arr.shape)
    Y = np.zeros(y_arr.shape)
    for idx, img in enumerate(x_arr):
        y_img = y_arr[idx,:,:,:]
        new_x, new_y = wiggle_image(img, y_img)
        X[idx, :,:,:] = new_x
        Y[idx,:,:,:] = new_y
    return X, Y


# In[9]:

def augment_train_val(x_train, y_train, x_val, y_val, aug_num = 10):

    x_train_aug = x_train.copy()
    y_train_aug = y_train.copy()

    x_val_aug = x_val.copy()
    y_val_aug = y_val.copy()


    for i in range(aug_num):
        x_train_a, y_train_a = augment_data(x_train, y_train)
        x_val_a, y_val_a = augment_data(x_val, y_val)

        x_train_aug = np.vstack((x_train_aug, x_train_a))
        y_train_aug = np.vstack((y_train_aug, y_train_a))
        x_val_aug = np.vstack((x_val_aug, x_val_a))
        y_val_aug = np.vstack((y_val_aug, y_val_a))
        print(x_train_aug.shape, x_val_aug.shape)
        #break
    return x_train_aug, y_train_aug, x_val_aug, y_val_aug


# In[10]:

def remove_hints(x_train_aug, x_val_aug):
    # randomly remove the hint in some images
    count = 0
    for i in range(x_train_aug.shape[0]):
        if np.random.binomial(1,0.1):
            count +=1
            x_train_aug[i,:,:,1] = 0

    print(count/x_train_aug.shape[0]*100, "% removed")

    count = 0
    for i in range(x_val_aug.shape[0]):
        if np.random.binomial(1,0.1):
            count +=1
            x_val_aug[i,:,:,1] = 0

    print(count/x_val_aug.shape[0]*100, "% removed")



# In[11]:

def weaken_hints(x_train_aug, x_val_aug):
    count = 0
    for i in range(x_train_aug.shape[0]):
        if np.random.binomial(1,0.3):
            count +=1
            hint = x_train_aug[i,:,:,1]
            x_train_aug[i,:,:,1], _ = wiggle_image(hint, hint)

    print(count/x_train_aug.shape[0]*100, "% wiggled hints")

    count = 0
    for i in range(x_val_aug.shape[0]):
        if np.random.binomial(1,0.3):
            count +=1
            hint = x_val_aug[i,:,:,1]
            x_val_aug[i,:,:,1], _ = wiggle_image(hint, hint)

    print(count/x_val_aug.shape[0]*100, "% wiggled hints")


# In[12]:

def dilate_or_erode_image(img, do_dilate):
    if do_dilate:
        return dilation(img, disk(3))
    else:
        return erosion(img, disk(3))


# In[13]:

def dilate_erode_hints(x_train_aug, x_val_aug):
    count = 0
    for i in range(x_train_aug.shape[0]):
        if np.random.binomial(1,0.3):
            count +=1
            hint = x_train_aug[i,:,:,1]
            x_train_aug[i,:,:,1] = dilate_or_erode_image(hint, np.random.binomial(1,0.5))

    print(count/x_train_aug.shape[0]*100, "% dilated or eroded hints")

    count = 0
    for i in range(x_val_aug.shape[0]):
        if np.random.binomial(1,0.3):
            count +=1
            hint = x_val_aug[i,:,:,1]
            x_val_aug[i,:,:,1] = dilate_or_erode_image(hint, np.random.binomial(1,0.5))

    print(count/x_val_aug.shape[0]*100, "% dilated or eroded hints")


# In[14]:

def get_new_log_dir():
    current_logs = sorted(glob("./log_try_????"))
    if len(current_logs) == 0:
        return "./log_try_0000"
    else:
        max_num = int(current_logs[-1].split("_")[-1])
        return "./log_try_%04d" % (max_num + 1)

def get_new_checkpoint_dir():
    current_logs = sorted(glob("./checkpoint_try_????"))
    if len(current_logs) == 0:
        return "./checkpoint_try_0000"
    else:
        max_num = int(current_logs[-1].split("_")[-1])
        return "./checkpoint_try_%04d" % (max_num + 1)


# In[15]:

def nearest_sq(n):
    root_of_n = np.sqrt(n)
    floor_integer = int(root_of_n)
    ceil_integer = floor_integer + 1
    floor_integer_square = floor_integer * floor_integer
    ceil_integer_square = ceil_integer * ceil_integer
    floor_distance = n - floor_integer_square
    ceil_distance = ceil_integer_square - n
    if floor_distance < ceil_distance:
        return np.sqrt(floor_integer_square)
    else:
        return np.sqrt(ceil_integer_square)


# In[16]:

#%matplotlib inline
def get_mosaics(save_path, base_images, truth_images, agg_images, comp_images):
    plt.style.use('dark_background')

    N = int(nearest_sq(len(base_images)))
    fig,axs = plt.subplots(N,N+1, figsize=(N,N+1))
    axs = axs.ravel()
    fig.subplots_adjust(wspace=0, hspace=0)
    [ax.axis("off") for ax in axs]

    for i, base in enumerate(base_images):
        t = truth_images[i]
        a = agg_images[i]

        #entry = {}
        #sub = base.split("/")[-2]
        #entry["sub"] = sub
        #s = int(base.split("/")[-1].replace("base","").split(".")[0])
        #entry["slice"] = s

        bdata = base #imread(base)
        tdata = truth_images[i] #imread(t).astype(np.float32)
        #entry["n_truth_vox"] = (tdata > 0).sum()
        tdata[tdata==0] = np.nan

        adata = agg_images[i] #(imread(a)).astype(np.float32)
        #entry["n_agg_vox"] = (adata > 0).sum()
        adata[adata < 1] = np.nan

        cdata = comp_images[i]
        cdata[cdata<0.1] = np.nan

        ax = axs[i]
        ax.imshow(bdata, cmap=plt.cm.Greys_r)
        ax.imshow(tdata, cmap=plt.cm.Reds, alpha=0.5, vmin=0, vmax=1)
        ax.imshow(adata, cmap=plt.cm.Blues, alpha=0.5, vmin=0, vmax=1)
        ax.imshow(cdata, alpha=0.5, vmin=0, vmax=1)
        ax.axis("off")

    plt.savefig(save_path)
    plt.close("all")


# In[ ]:


def run_everything(model_save_path, n_epochs=10, n_aug=10):

    stats = {}
    if not exists(model_save_path):
        makedirs(model_save_path)

    # get data
    images = sorted(glob("./ds_satra/tiles/*/base*.jpg"))
    hints = sorted(glob("./ds_satra/tiles/*/agg*.png"))
    masks =sorted(glob("./ds_satra/tiles/*/truth*.png"))
    assert(len(images) == len(masks))
    assert(len(hints) == len(masks))

    subjects_all = [i.split("/")[-2] for i in images]
    subjects = np.asarray(sorted(list(set(subjects_all))))

    bigM_base, bigM_mask = get_data(images, hints, masks)

    #splitting data
    train, test, val = get_split_indices(subjects, subjects_all)
    x_train = bigM_base[train, :]
    y_train = bigM_mask[train, :]

    x_test = bigM_base[test, :]
    y_test = bigM_mask[test, :]

    x_val = bigM_base[val, :]
    y_val = bigM_mask[val, :]

    #augment data
    x_train_aug, y_train_aug, x_val_aug, y_val_aug = augment_train_val(x_train, y_train, x_val, y_val, aug_num = n_aug)
    remove_hints(x_train_aug, x_val_aug)
    weaken_hints(x_train_aug, x_val_aug)
    dilate_erode_hints(x_train_aug, x_val_aug)

    #save all our data:
    np.savez(join(model_save_path, "data.npz"), **{"images": images, "hints": hints, "masks": masks,
                                                "subjects_all": subjects_all, "subjects": subjects,
                                                "bigM_base": bigM_base, "bigM_mask": bigM_mask,
                                                "train_idx": train, "test_idx": test, "val_idx": val,
                                                "x_train": x_train, "y_train": y_train, "x_val": x_val,
                                                "y_val": y_val, "x_test": x_test, "y_test": y_test,
                                                "x_train_aug": x_train_aug, "y_train_aug": y_train_aug,
                                                "x_val_aug": x_val_aug, "y_val_aug": y_val_aug})

    #run the model
    model = get_model()
    model.load_weights("pass03/test_ak_0027/model.h5")
    model.fit(x_train_aug, y_train_aug, batch_size=32,
          epochs=n_epochs, verbose=1, validation_data=(x_val_aug, y_val_aug),
          callbacks=[keras.callbacks.TensorBoard(log_dir=get_new_log_dir(), histogram_freq=0,
                                                 batch_size=32, write_graph=True,
                                                 write_grads=True, write_images=True,
                                                 embeddings_freq=0, embeddings_layer_names=None,
                                                 embeddings_metadata=None),
                     #keras.callbacks.ModelCheckpoint(get_new_checkpoint_dir(), monitor='val_dice',
                     #                                verbose=0, save_best_only=False, save_weights_only=False,
                     #                                mode='auto', period=1),
                     #keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0, patience=0, verbose=0, mode='auto'),
                     #keras.callbacks.EarlyStopping(monitor='val_dice', min_delta=0, patience=5, verbose=0, mode='max')
                    ]
    )

    score = model.evaluate(x_test, y_test)
    print("test score with hints", score)
    stats["score_with_hints"] = score

    x_test_no_hint = x_test.copy()
    x_test_no_hint[:,:,:,1] = 0
    score_no_hint = model.evaluate(x_test_no_hint, y_test)
    print("score w/ no hint", score_no_hint)
    stats["score w/ no hint"] = score_no_hint

    x_test_no_brain = x_test.copy()
    x_test_no_brain[:,:,:,0] = 0
    score_no_brain = model.evaluate(x_test_no_brain, y_test)
    print("score w/ no brain", score_no_brain)
    stats["score no brain"] = score_no_brain

    model.save("{}/model.h5".format(model_save_path))

    # save some prediciton images
    y_pred = model.predict(x_test)
    get_mosaics(join(model_save_path, "with_hint.png"), x_test[:,:,:,0], y_test[:,:,:,0],
               x_test[:,:,:,1], y_pred[:,:,:,0])


    y_pred_no_hint = model.predict(x_test_no_hint)
    get_mosaics(join(model_save_path, "without_hint.png"), x_test[:,:,:,0], y_test[:,:,:,0],
               x_test[:,:,:,1], y_pred_no_hint[:,:,:,0])


    y_pred_no_brain = model.predict(x_test_no_brain)
    get_mosaics(join(model_save_path, "without_brain.png"), x_test[:,:,:,0], y_test[:,:,:,0],
               x_test[:,:,:,1], y_pred_no_brain[:,:,:,0])

    stats["images"] = [join(model_save_path, "with_hint.png"),
                      join(model_save_path, "without_hint.png"),
                      join(model_save_path, "without_brain.png")]
    stats["n_epoch"] = n_epochs
    stats["n_aug"] = n_aug

    return stats



# In[ ]:

from subprocess import check_call, Popen, PIPE

if __name__ == "__main__":

    stats_all = []
    for i in range(10):
        stats = run_everything("test_ak_%04d" % i, 100, 5)
        stats_all.append(stats)
        save_json("model_stats.json", stats_all)
        cmds = ['bash', "gitcmd.sh", "%04d" % i]
        proc = Popen(cmds, stdout = PIPE)
        proc.wait()
        print(proc.stdout.readlines())
        print("completed iteration", i,"\n\n")



# In[ ]:
