//-*****************************************************************************
// Copyright 2015 Christopher Jon Horvath
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//-*****************************************************************************

//-*****************************************************************************
// This comes straight out of the GLFW docs, with a bit of reformatting
// by me. I'm also calling the UtilGL Init inside the GeepGLFW library.
//-*****************************************************************************

#include <GeepGLFW/All.h>
#include <stdlib.h>
#include <stdio.h>

static void error_callback(int error, const char* description) {
  fputs(description, stderr);
}

static void key_callback(GLFWwindow* window, int key, int scancode, int action,
                         int mods) {
  if (key == GLFW_KEY_ESCAPE && action == GLFW_PRESS) {
    glfwSetWindowShouldClose(window, GL_TRUE);
  }
}

int main(void) {
  GLFWwindow* window;
  glfwSetErrorCallback(error_callback);

  if (!glfwInit()) {
    exit(EXIT_FAILURE);
  }

  window = glfwCreateWindow(640, 480, "Simple example", NULL, NULL);
  if (!window) {
    glfwTerminate();
    exit(EXIT_FAILURE);
  }

  glfwMakeContextCurrent(window);

  // CJH: This is the only part I added.
  EncinoWaves::GeepGLFW::UtilGL::Init(true);

  glfwSetKeyCallback(window, key_callback);
  while (!glfwWindowShouldClose(window)) {
    float ratio;
    int width, height;
    glfwGetFramebufferSize(window, &width, &height);
    ratio = width / (float)height;
    glViewport(0, 0, width, height);
    glClear(GL_COLOR_BUFFER_BIT);
#if 0
        glMatrixMode( GL_PROJECTION );
        glLoadIdentity();
        glOrtho( -ratio, ratio, -1.f, 1.f, 1.f, -1.f );
        glMatrixMode( GL_MODELVIEW );
        glLoadIdentity();
        glRotatef( ( float ) glfwGetTime() * 50.f, 0.f, 0.f, 1.f );
        glBegin( GL_TRIANGLES );
        glColor3f( 1.f, 0.f, 0.f );
        glVertex3f( -0.6f, -0.4f, 0.f );
        glColor3f( 0.f, 1.f, 0.f );
        glVertex3f( 0.6f, -0.4f, 0.f );
        glColor3f( 0.f, 0.f, 1.f );
        glVertex3f( 0.f, 0.6f, 0.f );
        glEnd();
#endif
    glfwSwapBuffers(window);
    glfwPollEvents();
  }

  glfwDestroyWindow(window);
  glfwTerminate();
  exit(EXIT_SUCCESS);
}
