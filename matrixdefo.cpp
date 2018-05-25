
#include <igl/read_triangle_mesh.h>
#include <igl/viewer/Viewer.h>

#include <igl/readDMAT.h>
#include <igl/LinSpaced.h>
#include <igl/opengl/report_gl_error.h>

#include <Eigen/Core>

int main(int argc, char *argv[])
{
  using namespace Eigen;
  using namespace std;
  Eigen::MatrixXd V;
  Eigen::MatrixXi F;
  igl::read_triangle_mesh(argv[1],V,F);

  ///////////////////////////////////////////////////////////////////
  // Load and prepare data
  ///////////////////////////////////////////////////////////////////
  Eigen::Matrix< float,Eigen::Dynamic,1> I;
  Eigen::Matrix< float,Eigen::Dynamic,3,Eigen::RowMajor> tex;
  Eigen::Matrix< float,Eigen::Dynamic,Eigen::Dynamic,Eigen::RowMajor> U;
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


  ///////////////////////////////////////////////////////////////////
  // Initialize viewer and opengl context
  ///////////////////////////////////////////////////////////////////
  igl::viewer::Viewer v;
  v.data.set_mesh(V,F);
  v.data.set_face_based(false);
  v.core.show_lines = false;
  v.launch_init(true,false);
  v.opengl.shader_mesh.free();
  
  ///////////////////////////////////////////////////////////////////
  // Compile new shaders
  ///////////////////////////////////////////////////////////////////
  {
    std::string mesh_vertex_shader_string =
R"(#version 150
uniform mat4 model;
uniform mat4 view;
uniform mat4 proj;
in vec3 position;
in vec3 normal;
out vec3 position_eye;
out vec3 normal_eye;
in vec4 Ka;
in vec4 Kd;
in vec4 Ks;
in vec2 texcoord;
out vec2 texcoordi;
out vec4 Kai;
out vec4 Kdi;
out vec4 Ksi;

in float id;
uniform int n;
uniform int m;
uniform int s;
uniform float q[512];
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

  position_eye = vec3 (view * model * vec4 (deformed, 1.0));
  gl_Position = proj * vec4 (position_eye, 1.0);
  Kai = Ka;
  Kdi = Kd;
  Ksi = Ks;
})";

    std::string mesh_fragment_shader_string =
R"(#version 150
uniform mat4 model;
uniform mat4 view;
uniform mat4 proj;
uniform vec4 fixed_color;
in vec3 position_eye;
uniform vec3 light_position_world;
vec3 Ls = vec3 (1, 1, 1);
vec3 Ld = vec3 (1, 1, 1);
vec3 La = vec3 (1, 1, 1);
in vec4 Ksi;
in vec4 Kdi;
in vec4 Kai;
uniform float specular_exponent;
uniform float lighting_factor;
out vec4 outColor;
void main()
{
  vec3 xTangent = dFdx(position_eye);
  vec3 yTangent = dFdy(position_eye);
  vec3 normal_eye = normalize( cross( xTangent, yTangent ) );

vec3 Ia = La * vec3(Kai);    // ambient intensity
vec3 light_position_eye = vec3 (view * vec4 (light_position_world, 1.0));
vec3 vector_to_light_eye = light_position_eye - position_eye;
vec3 direction_to_light_eye = normalize (vector_to_light_eye);
float dot_prod = dot (direction_to_light_eye, normal_eye);
float clamped_dot_prod = max (dot_prod, 0.0);
vec3 Id = Ld * vec3(Kdi) * clamped_dot_prod;    // Diffuse intensity

vec3 reflection_eye = reflect (-direction_to_light_eye, normal_eye);
vec3 surface_to_viewer_eye = normalize (-position_eye);
float dot_prod_specular = dot (reflection_eye, surface_to_viewer_eye);
dot_prod_specular = float(abs(dot_prod)==dot_prod) * max (dot_prod_specular, 0.0);
float specular_factor = pow (dot_prod_specular, specular_exponent);
vec3 Kfi = 0.5*vec3(Ksi);
vec3 Lf = Ls;
float fresnel_exponent = 2*specular_exponent;
float fresnel_factor = 0;
{
  float NE = max( 0., dot( normal_eye, surface_to_viewer_eye));
  fresnel_factor = pow (max(sqrt(1. - NE*NE),0.0), fresnel_exponent);
}
vec3 Is = Ls * vec3(Ksi) * specular_factor;    // specular intensity
vec3 If = Lf * vec3(Kfi) * fresnel_factor;     // fresnel intensity
vec4 color = vec4(lighting_factor * (If + Is + Id) + Ia + 
  (1.0-lighting_factor) * vec3(Kdi),(Kai.a+Ksi.a+Kdi.a)/3);
outColor = color;
if (fixed_color != vec4(0.0)) outColor = fixed_color;
})";
    v.opengl.shader_mesh.init(
      mesh_vertex_shader_string,
      mesh_fragment_shader_string, 
      "outColor");
  }

  ///////////////////////////////////////////////////////////////////
  // Send texture and vertex attributes to GPU
  ///////////////////////////////////////////////////////////////////
  {
    GLuint prog_id = v.opengl.shader_mesh.program_shader;
    glUseProgram(prog_id);
    GLuint VAO = v.opengl.vao_mesh;
    glBindVertexArray(VAO);
    GLuint IBO;
    glGenBuffers(1, &IBO);
    glBindBuffer(GL_ARRAY_BUFFER, IBO);
    glBufferData(GL_ARRAY_BUFFER, sizeof(float)*I.size(), I.data(), GL_STATIC_DRAW);
    GLint iid = glGetAttribLocation(prog_id, "id");
    glVertexAttribPointer(
      iid, 1, GL_FLOAT, GL_FALSE, 1 * sizeof(GLfloat), (GLvoid*)0);
    glEnableVertexAttribArray(iid);
    glBindVertexArray(0);
    glActiveTexture(GL_TEXTURE0);
    //glGenTextures(1, &v.opengl.vbo_tex);
    glBindTexture(GL_TEXTURE_2D, v.opengl.vbo_tex);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glPixelStorei(GL_UNPACK_ALIGNMENT, 1);
    // 8650×8650 texture was roughly the max I could still get 60 fps, 8700²
    // already dropped to 1fps
    //
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, s,s, 0, GL_RGB, GL_FLOAT, tex.data());
  }


  Eigen::VectorXf q0 = Eigen::VectorXf::Zero(m,1);
  q0(0) = 1;
  Eigen::VectorXf q1 = Eigen::VectorXf::Zero(m,1);

  v.callback_pre_draw = [&U,&q0,&q1,&m,&n,&s](igl::viewer::Viewer & v) ->bool
  {
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
    /////////////////////////////////////////////////////////
    // Send uniforms to shader
    /////////////////////////////////////////////////////////
    GLuint prog_id = v.opengl.shader_mesh.program_shader;
    glUseProgram(prog_id);
    GLint n_loc = glGetUniformLocation(prog_id,"n");
    glUniform1i(n_loc,n);
    GLint m_loc = glGetUniformLocation(prog_id,"m");
    glUniform1i(m_loc,m);
    GLint s_loc = glGetUniformLocation(prog_id,"s");
    glUniform1i(s_loc,s);
    GLint q_loc = glGetUniformLocation(prog_id,"q");
    glUniform1fv(q_loc,U.cols(),qa.data());
    // Do this now so that we can stop texture from being loaded by viewer
    if (v.data.dirty)
    {
      v.opengl.set_data(v.data, v.core.invert_normals);
      v.data.dirty = igl::viewer::ViewerData::DIRTY_NONE;
    }
    v.opengl.dirty &= ~igl::viewer::ViewerData::DIRTY_TEXTURE;
    return false;
  };

  v.core.animation_max_fps = 60.0;
  v.core.is_animating = true;
  v.launch_rendering(true);
  v.launch_shut();
}
