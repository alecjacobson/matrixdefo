
#include <igl/read_triangle_mesh.h>
#include <igl/opengl/glfw/Viewer.h>
#include <igl/opengl/report_gl_error.h>
#include <igl/opengl/create_shader_program.h>
#include <igl/opengl/destroy_shader_program.h>

#include <igl/readDMAT.h>
#include <igl/LinSpaced.h>
#include <igl/opengl/destroy_shader_program.h>

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
  {
    Eigen::Matrix< double ,Eigen::Dynamic,Eigen::Dynamic,Eigen::RowMajor> Ud;
    igl::readDMAT(argv[2],Ud);
    U = Ud.cast<float>();
  }
  assert((U.rows() == V.rows()*3) && "#U should be 3*#V");
  //std::cout<<"**warning** resizing to min(U.cols(),100)"<<std::endl;
  //U.conservativeResize(U.rows(),std::min(100,(int)U.cols()));
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
        tex(i*m+j,c) = U(i+c*n,j);
      }
    }
  }


  ///////////////////////////////////////////////////////////////////
  // Initialize viewer and opengl context
  ///////////////////////////////////////////////////////////////////
  igl::opengl::glfw::Viewer v;
  v.data().set_mesh(V,F);
  v.data().set_face_based(false);
  v.data().show_lines = false;
  v.launch_init(true,false);
  v.data().meshgl.init();
  igl::opengl::destroy_shader_program(v.data().meshgl.shader_mesh);

  
  ///////////////////////////////////////////////////////////////////
  // Compile new shaders
  ///////////////////////////////////////////////////////////////////
  {
    std::string mesh_vertex_shader_string =
R"(#version 150
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
    int index = int(id)*m+j;
    int si = index % s;
    int sj = int((index - si)/s);
    displacement = displacement + texelFetch(tex,ivec2(si,sj),0).xyz*q[j];
  }
  vec3 deformed = position + displacement;

  position_eye = vec3 (view * vec4 (deformed, 1.0));
  gl_Position = proj * vec4 (position_eye, 1.0);
  Kai = Ka;
  Kdi = Kd;
  Ksi = Ks;
  texcoordi = texcoord;
})";

    std::string mesh_fragment_shader_string =
R"(#version 150
  uniform mat4 view;
  uniform mat4 proj;
  uniform vec4 fixed_color;
  in vec3 position_eye;
  uniform vec3 light_position_eye;
  vec3 Ls = vec3 (1, 1, 1);
  vec3 Ld = vec3 (1, 1, 1);
  vec3 La = vec3 (1, 1, 1);
  in vec4 Ksi;
  in vec4 Kdi;
  in vec4 Kai;
  in vec2 texcoordi;
  uniform sampler2D tex;
  uniform float specular_exponent;
  uniform float lighting_factor;
  uniform float texture_factor;
  uniform float matcap_factor;
  uniform float double_sided;
  out vec4 outColor;
  void main()
  {
    vec3 xTangent = dFdx(position_eye);
    vec3 yTangent = dFdy(position_eye);
    vec3 normal_eye = normalize( cross( yTangent, xTangent ) );

    if(matcap_factor == 1.0f)
    {
      vec2 uv = normalize(normal_eye).xy * 0.5 + 0.5;
      outColor = texture(tex, uv);
    }else
    {
      vec3 Ia = La * vec3(Kai);    // ambient intensity

      vec3 vector_to_light_eye = light_position_eye - position_eye;
      vec3 direction_to_light_eye = normalize (vector_to_light_eye);
      float dot_prod = dot (direction_to_light_eye, normalize(normal_eye));
      float clamped_dot_prod = abs(max (dot_prod, -double_sided));
      vec3 Id = Ld * vec3(Kdi) * clamped_dot_prod;    // Diffuse intensity

      vec3 reflection_eye = reflect (-direction_to_light_eye, normalize(normal_eye));
      vec3 surface_to_viewer_eye = normalize (-position_eye);
      float dot_prod_specular = dot (reflection_eye, surface_to_viewer_eye);
      dot_prod_specular = float(abs(dot_prod)==dot_prod) * abs(max (dot_prod_specular, -double_sided));
      float specular_factor = pow (dot_prod_specular, specular_exponent);
      vec3 Is = Ls * vec3(Ksi) * specular_factor;    // specular intensity
      vec4 color = vec4(lighting_factor * (Is + Id) + Ia + (1.0-lighting_factor) * vec3(Kdi),(Kai.a+Ksi.a+Kdi.a)/3);
      outColor = mix(vec4(1,1,1,1), texture(tex, texcoordi), texture_factor) * color;
      if (fixed_color != vec4(0.0)) outColor = fixed_color;
    }
  }
)";

    igl::opengl::create_shader_program(
      mesh_vertex_shader_string,
      mesh_fragment_shader_string,
      {},
      v.data().meshgl.shader_mesh);
  }

  ///////////////////////////////////////////////////////////////////
  // Send texture and vertex attributes to GPU
  ///////////////////////////////////////////////////////////////////
  {
    GLuint prog_id = v.data().meshgl.shader_mesh;
    glUseProgram(prog_id);
    GLuint VAO = v.data().meshgl.vao_mesh;
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
    //glGenTextures(1, &v.data().meshgl.vbo_tex);
    glBindTexture(GL_TEXTURE_2D, v.data().meshgl.vbo_tex);
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

  v.callback_pre_draw = [&U,&q0,&q1,&m,&n,&s](igl::opengl::glfw::Viewer & v) ->bool
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
    GLuint prog_id = v.data().meshgl.shader_mesh;
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
    if (v.data().dirty)
    {
      v.data().updateGL(
        v.data(), 
        v.data().invert_normals,
        v.data().meshgl
        );
      v.data().dirty = igl::opengl::MeshGL::DIRTY_NONE;
    }
    v.data().meshgl.dirty &= ~igl::opengl::MeshGL::DIRTY_TEXTURE;
    return false;
  };

  v.core().animation_max_fps = 60.0;
  v.core().is_animating = true;
  v.launch_rendering(true);
  v.launch_shut();
}
