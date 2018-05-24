// make sure the modern opengl headers are included before any others
#include <OpenGL/gl3.h>
#define __gl_h_

#include <igl/frustum.h>
#include <igl/get_seconds.h>
// WTF, why do I have to do this. Glad is fucking things up on mac
#define IGL_OPENGL_GL_H
#include <igl/opengl/create_shader_program.h>
#include <igl/opengl/report_gl_error.h>
#include <igl/readDMAT.h>
#include <igl/read_triangle_mesh.h>
#include <igl/get_seconds.h>
#include <Eigen/Core>
#define GLFW_INCLUDE_GLU
#include <GLFW/glfw3.h>

#include <string>
#include <chrono>
#include <thread>
#include <iostream>

#ifndef NDEBUG
  #define GL_CHECK(stmt) \
    do \
    { \
      stmt; \
      GLenum err = glGetError(); \
      if (err != GL_NO_ERROR) \
      { \
        const auto gluErrorString = [](GLenum errorCode)->const char * \
        { \
          switch(errorCode) \
          { \
            default: \
              return "unknown error code"; \
            case GL_NO_ERROR: \
              return "no error"; \
            case GL_INVALID_ENUM: \
              return "invalid enumerant"; \
            case GL_INVALID_VALUE: \
              return "invalid value"; \
            case GL_INVALID_OPERATION: \
              return "invalid operation"; \
            case GL_OUT_OF_MEMORY: \
              return "out of memory"; \
            case GL_INVALID_FRAMEBUFFER_OPERATION: \
              return "invalid framebuffer operation"; \
          } \
        }; \
        printf( \
         "OpenGL error %08x (%s), at %s:%i - for %s\n", \
         err, gluErrorString(err), __FILE__, __LINE__, #stmt); \
      } \
    } while (0)
  // Should I just have _all_ opengl functions wrapped like this?
#  define    glVertexAttribPointer(X1,X2,X3,X4,X5,X6) \
    GL_CHECK(glVertexAttribPointer(X1,X2,X3,X4,X5,X6))
#  define    glTexImage2D(X1,X2,X3,X4,X5,X6,X7,X8,X9) \
    GL_CHECK(glTexImage2D(X1,X2,X3,X4,X5,X6,X7,X8,X9))
#else
  #define GL_CHECK(stmt) stmt
#endif

std::string vertex_shader = R"(
#version 330 core
uniform mat4 proj;
uniform mat4 model;
uniform int n;
uniform int m;
uniform int s;
uniform float q[512];
in vec3 position;
in float id;
out vec3 vcolor;
out vec4 vmodel_position;
uniform sampler2D tex;
void main()
{
  vec3 displacement = vec3(0,0,0);
  for(int j = 0;j < m; j++)
  {
    int index = int(id)+j*n;
    int si = index % s;
    int sj = int((index - si)/s);
    displacement = displacement + texelFetch(tex,ivec2(si,sj),0).xyz*q[j];
  }
  vec3 deformed = position + displacement;
  vmodel_position = model * vec4(deformed,1.);
  gl_Position = proj * vmodel_position;
  //vcolor = vec3(0.8,0.3,0.1);

  int j = 0;
  int index = int(id)+j*n;
  int si = index % s;
  int sj = int((index - si)/s);
  //vcolor = texelFetch(tex,ivec2(si,sj),0).xyz/0.07249*0.5+0.5;
  vcolor = (displacement*10.0*0.5+0.5);
}
)";
std::string fragment_shader = R"(
#version 330 core
in vec3 vcolor;
in vec4 vmodel_position;
out vec3 color;
void main()
{
  vec3 xTangent = dFdx(vmodel_position.xyz);
  vec3 yTangent = dFdy(vmodel_position.xyz);
  vec3 normal = normalize( cross( xTangent, yTangent ) );
  color = max(normal.z,0.0)*vcolor;
}
)";

// width, height, shader id, vertex array object
int w=800,h=600;
double highdpi=1;
GLuint prog_id=0;
GLuint VAO;
// Mesh data: RowMajor is important to directly use in OpenGL
Eigen::Matrix< float,Eigen::Dynamic,3,Eigen::RowMajor> V;
Eigen::Matrix< float,Eigen::Dynamic,1> I;
Eigen::Matrix< float,Eigen::Dynamic,3,Eigen::RowMajor> tex;
Eigen::Matrix< float,Eigen::Dynamic,Eigen::Dynamic,Eigen::RowMajor> U;
Eigen::Matrix<GLuint,Eigen::Dynamic,3,Eigen::RowMajor> F;
int main(int argc, char * argv[])
{
  using namespace std;
  if(!glfwInit())
  {
     cerr<<"Could not initialize glfw"<<endl;
     return EXIT_FAILURE;
  }
  const auto & error = [] (int error, const char* description)
  {
    cerr<<description<<endl;
  };
  glfwSetErrorCallback(error);
  glfwWindowHint(GLFW_SAMPLES, 4);
  glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
  glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 2);
  glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
  glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE);
  GLFWwindow* window = glfwCreateWindow(w, h, "WebGL", NULL, NULL);
  if(!window)
  {
    glfwTerminate();
    cerr<<"Could not create glfw window"<<endl;
    return EXIT_FAILURE;
  }

  glfwMakeContextCurrent(window);

      int major, minor, rev;
      major = glfwGetWindowAttrib(window, GLFW_CONTEXT_VERSION_MAJOR);
      minor = glfwGetWindowAttrib(window, GLFW_CONTEXT_VERSION_MINOR);
      rev = glfwGetWindowAttrib(window, GLFW_CONTEXT_REVISION);
      printf("OpenGL version recieved: %d.%d.%d\n", major, minor, rev);
      printf("Supported OpenGL is %s\n", (const char*)glGetString(GL_VERSION));
      printf("Supported GLSL is %s\n", (const char*)glGetString(GL_SHADING_LANGUAGE_VERSION));
      GLint max_tex;
      glGetIntegerv(GL_MAX_TEXTURE_SIZE,&max_tex);
      printf("GL_MAX_TEXTURE_SIZE %d\n",max_tex);

  glfwSetInputMode(window,GLFW_CURSOR,GLFW_CURSOR_NORMAL);
  const auto & reshape = [] (GLFWwindow* window, int w, int h)
  {
    ::w=w,::h=h;
  };
  glfwSetWindowSizeCallback(window,reshape);

  {
    int width, height;
    glfwGetFramebufferSize(window, &width, &height);
    int width_window, height_window;
    glfwGetWindowSize(window, &width_window, &height_window);
    highdpi = width/width_window;
    reshape(window,width_window*highdpi,height_window*highdpi);
  }

  // Compile each shader
  igl::opengl::create_shader_program(vertex_shader,fragment_shader,{},prog_id);

  // Read input mesh from file
  igl::read_triangle_mesh(argv[1],V,F);

  // Assume that input is already registered to unit sphere
  //V.rowwise() -= V.colwise().mean();
  //V /= (V.colwise().maxCoeff()-V.colwise().minCoeff()).maxCoeff();
  //V /= 2.2;

  // U = [Ux;Uy;Uz], where Ux ∈ R^(n×k)
  igl::readDMAT(argv[2],U);
  assert((U.rows() == V.rows()*3) && "#U should be 3*#V");
  std::cout<<"**warning** resizing to min(U.cols(),100)"<<std::endl;
  U.conservativeResize(U.rows(),std::min(100,(int)U.cols()));

  I = igl::LinSpaced< Eigen::Matrix< float,Eigen::Dynamic,1> >(V.rows(),0,V.rows()-1);

  const int n = V.rows();
  const int m = U.cols();
  const int s = ceil(sqrt(n*m));
  assert(s*s > n*m);
  printf("%d %d %d\n",n,m,s);
  tex = Eigen::Matrix< float,Eigen::Dynamic,3,Eigen::RowMajor>::Zero(s*s,3);
  for(int j = 0;j<m;j++)
  {
    for(int i = 0;i<n;i++)
    {
      for(int c = 0;c<3;c++)
      {
        tex(i+j*n,c) = U(i+c*n,j);
      }
    }
  }

  // Generate and attach buffers to vertex array
  glGenVertexArrays(1, &VAO);
  GLuint VBO, NBO, IBO, CBO, EBO;
  glGenBuffers(1, &VBO);
  glGenBuffers(1, &IBO);
  glGenBuffers(1, &EBO);
  glBindVertexArray(VAO);
      igl::opengl::report_gl_error("7: ");

  glBindBuffer(GL_ARRAY_BUFFER, VBO);
  glBufferData(GL_ARRAY_BUFFER, sizeof(float)*V.size(), V.data(), GL_STATIC_DRAW);
  GLint pid = glGetAttribLocation(prog_id, "position");
  glVertexAttribPointer(
    pid, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(GLfloat), (GLvoid*)0);
  glEnableVertexAttribArray(pid);
      igl::opengl::report_gl_error("6: ");

  glBindBuffer(GL_ARRAY_BUFFER, IBO);
  glBufferData(GL_ARRAY_BUFFER, sizeof(float)*I.size(), I.data(), GL_STATIC_DRAW);
  GLint iid = glGetAttribLocation(prog_id, "id");
  glVertexAttribPointer(
    iid, 1, GL_FLOAT, GL_FALSE, 1 * sizeof(GLfloat), (GLvoid*)0);
  glEnableVertexAttribArray(iid);
      igl::opengl::report_gl_error("5: ");

  glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, EBO);
  glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(GLuint)*F.size(), F.data(), GL_STATIC_DRAW);
  glBindBuffer(GL_ARRAY_BUFFER, 0); 
  glBindVertexArray(0);
      igl::opengl::report_gl_error("3: ");

  glActiveTexture(GL_TEXTURE0);
  GLuint vbo_tex;
  glGenTextures(1, &vbo_tex);

  glBindTexture(GL_TEXTURE_2D, vbo_tex);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
  glPixelStorei(GL_UNPACK_ALIGNMENT, 1);
  // 8650×8650 texture was roughly the max I could still get 60 fps, 8700²
  // already dropped to 1fps
  //
  glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, s,s, 0, GL_RGB, GL_FLOAT, tex.data());
  //{
  //  const int s = 8650;
  //  Eigen::VectorXf tex = Eigen::VectorXf::Zero(s*s*3);
  //  glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, s,s, 0, GL_RGB, GL_FLOAT, tex.data());
  //}
  igl::opengl::report_gl_error("2: ");


  // Main display routine
  Eigen::VectorXf q0 = Eigen::VectorXf::Zero(m,1);
  q0(0) = 1;
  Eigen::VectorXf q1 = Eigen::VectorXf::Zero(m,1);
  while (!glfwWindowShouldClose(window))
  {
      double tic = igl::get_seconds();
      // clear screen and set viewport
      glClearColor(0.0,0.4,0.7,0.);
      glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
      glViewport(0,0,w,h);
      glEnable(GL_DEPTH_TEST);

      // Projection and modelview matrices
      Eigen::Matrix4f proj = Eigen::Matrix4f::Identity();
      float near = 0.01;
      float far = 100;
      float top = tan(35./360.*M_PI)*near;
      float right = top * (double)::w/(double)::h;
      igl::frustum(-right,right,-top,top,near,far,proj);
      Eigen::Affine3f model = Eigen::Affine3f::Identity();
      model.translate(Eigen::Vector3f(0,0,-1.5));
      // spin around
      static size_t count = 0;
      const int keyrate = 15;
      if(count % keyrate == 0)
      {
        q0 = q1;
        q1 = Eigen::VectorXf::Random(m,1).array()*0.5+0.5;
        q1 = q1.array().pow(100.0).eval();
      }
      Eigen::VectorXf qa = q0 + double(count % keyrate)/(keyrate-1.0) * (q1 - q0);
      qa /= qa.sum();
      count++;

      static size_t fps_tic_count = 0;
      fps_tic_count++;
      static double fps_tic = igl::get_seconds();
      model.rotate(Eigen::AngleAxisf(M_PI,Eigen::Vector3f(0,1,0)));
      model.rotate(Eigen::AngleAxisf(-M_PI*0.5,Eigen::Vector3f(0,0,1)));
      {
        const double fps_toc = igl::get_seconds();
        if((fps_toc - fps_tic) > 2.0)
        {
          std::cout<<double(fps_tic_count)/(fps_toc-fps_tic)<<" fps"<<std::endl;
          fps_tic = igl::get_seconds();
          fps_tic_count = 0;
        }
      }

      // select program and attach uniforms
      igl::opengl::report_gl_error("1: ");
      glUseProgram(prog_id);
      GLint proj_loc = glGetUniformLocation(prog_id,"proj");
      glUniformMatrix4fv(proj_loc,1,GL_FALSE,proj.data());
      GLint model_loc = glGetUniformLocation(prog_id,"model");
      glUniformMatrix4fv(model_loc,1,GL_FALSE,model.matrix().data());
      GLint n_loc = glGetUniformLocation(prog_id,"n");
      glUniform1i(n_loc,n);
      GLint m_loc = glGetUniformLocation(prog_id,"m");
      glUniform1i(m_loc,m);
      GLint s_loc = glGetUniformLocation(prog_id,"s");
      glUniform1i(s_loc,s);
      GLint q_loc = glGetUniformLocation(prog_id,"q");
      igl::opengl::report_gl_error("0: ");
      glUniform1fv(q_loc,U.cols(),qa.data());
      igl::opengl::report_gl_error("after: ");

      // Draw mesh as wireframe
      glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
      glBindVertexArray(VAO);
      glActiveTexture(GL_TEXTURE0);
      glBindTexture(GL_TEXTURE_2D, vbo_tex);
      glDrawElements(GL_TRIANGLES, F.size(), GL_UNSIGNED_INT, 0);
      glBindVertexArray(0);

      glfwSwapBuffers(window);

      {
        glfwPollEvents();
        // In microseconds
        double duration = 1000000.*(igl::get_seconds()-tic);
        const double min_duration = 1000000./60.;
        if(duration<min_duration)
        {
          std::this_thread::sleep_for(std::chrono::microseconds((int)(min_duration-duration)));
        }
      }
  }
  glfwDestroyWindow(window);
  glfwTerminate();
  return EXIT_SUCCESS;
}
