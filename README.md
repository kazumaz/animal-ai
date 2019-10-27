### animal-ai
image recognition ai (web service using FLASK)
 
### how to use
* put pictures to each anmimal directories `boar, crow, monkey`
* start preprocessing data 

execute command `python gen_data_augmented.py`  

output `animal_aug.npy`
* start learning 

execute command `python anim al_cnn_aug.py` 

output `animal_cnn_aug.hs` 
* start image recognition service 

execute command `python -m flask run --host=0.0.0.0 --port=5000 --without-threads` 
* access [image recognition service](http://localhost:5000/) and check the result 
 
