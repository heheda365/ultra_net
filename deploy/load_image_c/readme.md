## Use c code to load the images
- command:  g++ -shared -O2 load_image.cpp -o load_image.so -fPIC `pkg-config opencv --cflags --libs` -lpthread