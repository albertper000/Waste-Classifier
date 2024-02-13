import numpy as np
import os
from kivy.app import App
from PIL import Image
from kivy.lang import Builder
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.button import Button
import time
from kivy.uix.camera import Camera
from kivy.uix.label import Label
from kivy.uix.button import Button
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.togglebutton import ToggleButton
import tensorflow as tf

waste = 'class'
prediction = 'confidence'


class cameraApp(App):
    
    def build(self):
       
        layout = BoxLayout(orientation='vertical')

        # create camera instance
        self.cam = Camera(resolution = (640, 480))
        self.cam = Camera(play=True)
        

        # create button
        self.btn = Button(
            text="Capture",
            size_hint=(1, 0.2),
            font_size=35,
            background_color='green',
            on_press=self.capture)

        self.lbl_class = Label(
            text=waste,
            size_hint=(1, 0.2))

        self.lbl_conf = Label(
            text=prediction,
            size_hint=(1, 0.2))

        # add widgets in layout
        layout.add_widget(self.cam)
        layout.add_widget(self.btn)
        layout.add_widget(self.lbl_class)
        layout.add_widget(self.lbl_conf)

        return layout

    def capture(self, *args):
        self.cam.export_to_png(os.path.join(os.getcwd(), 'img.png'))
        waste, prediction = self.predict()
        self.lbl_class.text = waste
        self.lbl_conf.text = prediction
        
    def predict(self, model='modelOrdenado30.tflite'):
        waste = [
           'Battery',
           'Biological',
           'Cardboard Paper',
           'Glass',
           'Plastic Metal']
        
        interpreter = tf.lite.Interpreter(model_path='modelOrdenado30.tflite')
        interpreter.allocate_tensors()

        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()

        img = Image.open(os.path.join(os.getcwd(), 'img.png'))
        img_arr = np.array(img.resize((224, 224)), np.float32)
        img_arr = img_arr[:, :, :3]
        img_arr = np.expand_dims(img_arr, axis=0)
        
        interpreter.set_tensor(input_details[0]['index'], img_arr)

        interpreter.invoke()

        #resultado = dict(zip(waste, list(interpreter.get_tensor(output_details[0]['index']))))
        #best = max(resultado, key=resultado.get) 

        resultado = interpreter.get_tensor(output_details[0]['index'])

        best = resultado.argmax()

        waste[best]
        prediction = str(round(resultado.max()*100, 2))+"%"

        return waste[best], prediction


cameraApp().run()


