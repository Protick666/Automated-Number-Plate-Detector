from tkinter import filedialog
from tkinter import *
from PIL import ImageTk, Image
from train import NN
import cv2
import numpy as np
from Lib.ocr import DataSet
import os
import tensorflow as tf
from keras import backend as K
from Lib.ocr import decode_batch
import shutil

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

sess = tf.Session()
K.set_session(sess)

print('Loading Neural Network')
nn = NN()
nn.loadHyperParam('hyperparameters.txt')
nn.load_saved_weight()

letters = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'dhaka', 'metro', 'ka', 'kha', 'ga', 'gha', 'cho', 'so',
           'mo', 'na', 'bo',
           'chatto', 'ha', 'chandpur', 'khulna', 'jo']

root = Tk()
root.title('License Plate Detection')
var = StringVar()
frame = Frame(root, height=800, width=900)
frame.pack_propagate(0)
frame.pack()
label = Label(frame, textvariable=var)
var.set('Please Select an image or video to detect license plate')
label.pack()

label1 = None

image_path = None


def detectImage():
    global label1, image_path
    cwd = os.getcwd()
    temp = os.path.join(cwd, 'Temp')
    if not os.path.isdir(temp):
        os.mkdir(temp)
    image = cv2.imread(image_path)
    image_h, image_w, _ = image.shape
    boxes = nn.yolo.predict(image)
    index = 0
    annotation = []
    print(len(boxes), 'boxes found')
    default_label = '0-0-0-0-0-0-0-0-0'
    for box in boxes:
        xmin = int(box.xmin * image_w)
        ymin = int(box.ymin * image_h)
        xmax = int(box.xmax * image_w)
        ymax = int(box.ymax * image_h)
        num_plate = image[ymin:ymax, xmin:xmax]
        r = num_plate.shape[0]
        c = num_plate.shape[1]

        s = int(r * .6)
        d = r - s
        up = num_plate[:s, :]
        down = num_plate[d:, :]

        num_plate = np.concatenate((up, down), axis=1)
        cv2.imwrite('Temp/' + str(index) + '.jpg', num_plate)
        annotation.append(['Temp/' + str(index) + '.jpg', default_label])

    annotation = np.array(annotation)
    model = nn.ocr.model
    net_inp = model.get_layer(name='the_input').input
    net_out = model.get_layer(name='softmax').output

    testDS = DataSet('./', annotation, 128, 64, 8, 4)
    testDS.build_data()

    for inp_value, _ in testDS.next_batch():
        bs = inp_value['the_input'].shape[0]
        X_data = inp_value['the_input']
        net_out_value = sess.run(net_out, feed_dict={net_inp: X_data})
        pred_texts = decode_batch(net_out_value)
        labels = inp_value['the_labels']
        texts = []
        for label in labels:
            text = ' '.join(list(map(lambda x: letters[int(x)], label)))
            texts.append(text)

        for i in range(bs):
            # print('Predicted: %s\n' % (pred_texts[i]))
            image_h, image_w, _ = image.shape
            xmin = int(boxes[i].xmin * image_w)
            ymin = int(boxes[i].ymin * image_h)
            xmax = int(boxes[i].xmax * image_w)
            ymax = int(boxes[i].ymax * image_h)

            cv2.rectangle(image, (xmin, ymin), (xmax, ymax), (0, 255, 0), 3)
            cv2.putText(image,
                        pred_texts[i],
                        (xmin, ymin - 13),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1e-3 * image_h,
                        (0, 0, 255), 2)
            break

        cv2.imwrite(image_path[:-4] + '_detected' + image_path[-4:], image)
        im = Image.open(image_path[:-4] + '_detected' + image_path[-4:])
        im.thumbnail((400, 400), Image.ANTIALIAS)
        photo = ImageTk.PhotoImage(im)
        label12 = Label(frame, image=photo)
        label12.pack()
        label12.place(x=450, y=100)
        label12.img = photo
        var1 = StringVar()
        label3 = Label(frame, textvariable=var1, )

        var1.set(pred_texts[i])
        label3.pack()
        label3.place(x=500, y=550)
        return


def openImageFileButton():
    global label1, image_path
    filename = filedialog.askopenfilename(initialdir="/", title="Select jpg,png or mp4 file")
    print(filename)
    im = Image.open(filename)
    im.thumbnail((400, 400), Image.ANTIALIAS)
    photo = ImageTk.PhotoImage(im)
    label1 = Label(frame, image=photo)

    label1.pack()
    label1.place(x=10, y=100)
    label1.img = photo
    image_path = filename

    detectBt = Button(frame, text='Find License Plate On This Image', command=detectImage)

    detectBt.pack()
    detectBt.place(x=10, y=550)


openImageFileBt = Button(frame, text='Open Image', command=openImageFileButton)
openImageFileBt.pack()
openImageFileBt.place(x=50, y=50)

openVideoFileBt = Button(frame, text='Open Image', command=openImageFileButton)
openImageFileBt.pack()
openImageFileBt.place(x=50, y=50)

root.mainloop()
shutil.rmtree('Temp')
