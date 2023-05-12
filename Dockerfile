FROM gcc:latest

WORKDIR /home/

CMD ["/bin/bash"]

# docker build -t gcc .
# docker run -it --rm -v ${PWD}:/home/ gcc

# gcc -o main -lpthread main.c && ./main
# or just make it