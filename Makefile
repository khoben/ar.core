CXX		  := g++
CXX_FLAGS := -Wall -Wextra -std=c++17 -ggdb

OPENCV_LIB := "D:\opencv-4.1.1\build\install\x64\mingw\lib"
OPENCV_INC := "D:\opencv-4.1.1\build\install\include"
OPENCV_RUNTIME_LIB := "D:\opencv-4.1.1\build\install\x64\mingw\bin"

BIN		:= bin
SRC		:= src
INCLUDE	:= include $(OPENCV_INC)
LIB		:= lib $(OPENCV_LIB)

INCLUDE_PARAM := $(addprefix -I,$(INCLUDE))
LIB_PARAM := $(addprefix -L,$(LIB))

LIBRARIES	:= -llibopencv_core411 -llibopencv_imgcodecs411 -llibopencv_highgui411 \
 -llibopencv_imgproc411 -llibopencv_videoio411 -llibopencv_video411


EXECUTABLE	:= main

ifeq ($(OS),Windows_NT)
    CLEAR = cls
	RM = del /Q
	BRING_DLL = xcopy lib bin
else
    CLEAR = clear
	RM = rm -f
endif

all: $(BIN)/$(EXECUTABLE)

run: clean all
	$(BRING_DLL)
	$(CLEAR)
	./$(BIN)/$(EXECUTABLE)

$(BIN)/$(EXECUTABLE): $(SRC)/*.cpp
	$(CXX) $(CXX_FLAGS) $(INCLUDE_PARAM) $(LIB_PARAM) $^ -o $@ $(LIBRARIES) 

clean:
	-$(RM) $(BIN)
