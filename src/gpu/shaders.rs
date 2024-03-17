pub mod cs {
    vulkano_shaders::shader! {
        ty: "compute",
        src: r"
            #version 460

            layout(local_size_x = 8, local_size_y = 8, local_size_z = 1) in;

            layout(set = 0, binding = 0, rgba8) uniform writeonly image2D img;
            layout(set = 0, binding = 1) uniform LaunchInfo {
                vec3 cam_pos;
                float fov;
                mat4 lookat;
            } launch_info;

            const int MAX_MARCHING_STEPS = 1000;
            const float MIN_DIST = 0.0;
            const float MAX_DIST = 1000.0;
            const float EPSILON = 0.0001;

            float sphere_sdf(vec3 p) {
                return length(p) - 1.0;
            }

            struct ObjInfo {
                float sdf;
                vec3 ambient;
                vec3 diffuse;
                vec3 specular;
                float shininess;
            };

            mat4 lookat_matrix(vec3 cam_pos, vec3 center, vec3 up) {
                vec3 f = normalize(center - cam_pos);
                vec3 s = normalize(cross(f, up));
                vec3 u = cross(s, f);

                return mat4(
                    vec4(s, 0.0),
                    vec4(u, 0.0),
                    vec4(-f, 0.0),
                    vec4(0.0, 0.0, 0.0, 1)
                );
            }

            ObjInfo union_sdf(ObjInfo first, ObjInfo second) {
                if (first.sdf < second.sdf) {
                    return first;
                }

                return second;
            }

            float cube_sdf(vec3 p, vec3 b) {
                vec3 d = abs(p) - b;
                
                float inside_dist = min(max(max(d.y, d.z), d.x), 0.0);

                float outside_dist = length(max(vec3(0.0, 0.0, 0.0), d));

                return outside_dist + inside_dist;
            }
            
            float menger_sdf(vec3 p) {
                float d = cube_sdf(p, vec3(1.0));

                float s = 1.0;

                for(int m = 0; m < 5; m++) {
                    vec3 a = mod (p*s, 2.0) - 1.0;
                    s *= 3.0;
                    vec3 r = abs(1.0 - 3.0 * abs(a));

                    float da = max(r.x, r.y);
                    float db = max(r.y, r.z);
                    float dc = max(r.z, r.x);
                    float c = (min(da, min(db, dc)) - 1.0) / s;

                    d = max(d, c);
                }

                return d;
            }

            float plain_sdf(vec3 p) {
                return p.y;
            }

            ObjInfo scene_sdf(vec3 p) {
                return union_sdf(
                    ObjInfo(
                        menger_sdf(p - vec3(0.0, 1.0, 0.0)),
                        vec3(0.5, 0.5, 0.5),
                        vec3(0.5, 0.5, 0.5),
                        vec3(2.0, 2.0, 2.0),
                        15.0
                    ),
                    union_sdf(
                        ObjInfo(
                            sphere_sdf(p - vec3(3.0, 1.0, 0.0)),
                            vec3(0.5, 0.2, 0.2),
                            vec3(0.6, 0.2, 0.2),
                            vec3(2.0, 2.0, 2.0),
                            15.0
                        ),
                        ObjInfo(
                            plain_sdf(p - vec3(0.0, 0.0, 0.0)),
                            vec3(1.1, 1.2, 1.0),
                            vec3(0.6, 0.1, 0.65),
                            vec3(1.0, 1.0, 1.0),
                            20.0
                        )
                    )
                );
            }

            vec3 cam_dir (float fov, vec2 size, vec2 frag_coord) {
                vec2 xy = frag_coord - size / 2.0;
                float z = size.y / tan(radians(fov) / 2.0);
                return normalize(vec3(xy.x, -xy.y, -z));
            }

            vec3 estimate_normal(vec3 p) {
                return normalize( vec3(
                    scene_sdf(vec3(p.x + EPSILON, p.y, p.z)).sdf - scene_sdf(vec3(p.x - EPSILON, p.y, p.z)).sdf,
                    scene_sdf(vec3(p.x, p.y + EPSILON, p.z)).sdf - scene_sdf(vec3(p.x, p.y - EPSILON, p.z)).sdf,
                    scene_sdf(vec3(p.x, p.y, p.z + EPSILON)).sdf - scene_sdf(vec3(p.x, p.y, p.z - EPSILON)).sdf
                ));
            }

            vec3 phong_contrib_for_light(vec3 diffuse, vec3 specular, float alpha, vec3 p, vec3 cam_pos, vec3 light_pos, vec3 light_intensity) {
                vec3 normal = estimate_normal(p);
                vec3 light = normalize(light_pos - p);
                vec3 view = normalize(cam_pos - p);
                vec3 r = normalize(reflect(-light, normal));

                float dotln = dot(light, normal);
                float dotrv = clamp(dot(r, view), 0.0, 1.0);

                if (dotln < 0.0) {
                    return vec3(0.0, 0.0, 0.0);
                }

                vec3 res = light_intensity * diffuse * dotln + specular * pow(dotrv, alpha);

                vec3 sha_start = p + normal * 0.1;

                vec3 light_dir = normalize(light_pos - sha_start);

                float depth = MIN_DIST;

                for (int i = 0; i < MAX_MARCHING_STEPS; i++) {
                    ObjInfo obj = scene_sdf(sha_start + light_dir * depth);

                    if (obj.sdf < 0.01) {
                        res *= 0.0;

                        break;
                    }

                    depth += obj.sdf;

                    if (depth > MAX_DIST - EPSILON) {
                        break;
                    }
                }

                return res;
            }

            vec3 phong_illumination(vec3 ambient, vec3 diffuse, vec3 specular, float alpha, vec3 p, vec3 cam_pos) {
                vec3 ambient_light = vec3(1.0, 1.0, 1.0) * 0.2;

                vec3 color = ambient * ambient_light;

                vec3 light_pos = vec3(0.0, 4.0, 3.0);

                vec3 light_intensity = vec3(0.7, 0.7, 0.7);

                color += phong_contrib_for_light(diffuse, specular, alpha, p, cam_pos, light_pos, light_intensity);

                return color;
            }

            vec3 render(vec3 cam_pos, vec3 march_dir, float start, float end) {
                float depth = start;

                for (int i = 0; i < MAX_MARCHING_STEPS; i++) {
                    ObjInfo obj = scene_sdf(cam_pos + march_dir * depth);

                    if (obj.sdf < EPSILON) {
                        return phong_illumination(
                            obj.ambient,
                            obj.diffuse,
                            obj.specular,
                            obj.shininess,
                            cam_pos + march_dir * depth,
                            cam_pos
                        );
                    }

                    depth += obj.sdf;

                    if (depth >= end - EPSILON) {
                        return vec3(170.0 / 255.0, 150.0 / 255.0, 1.0);
                    }
                }

                return vec3(170.0 / 255.0, 150.0 / 255.0, 1.0);
            }

            void main() {
                float fov = launch_info.fov;
                vec3 cam_pos = launch_info.cam_pos;

                vec3 view_dir = cam_dir(fov, imageSize(img).xy, gl_GlobalInvocationID.xy);
                vec3 world_dir = (launch_info.lookat * vec4(view_dir, 0.0)).xyz;

                vec3 color = render(cam_pos, world_dir, MIN_DIST, MAX_DIST);

                imageStore(img, ivec2(gl_GlobalInvocationID.xy), vec4(color, 1.0));
                
            }
        "
    }
}