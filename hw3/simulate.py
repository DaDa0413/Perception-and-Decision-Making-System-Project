from os import minor
import numpy as np
from PIL import Image
import numpy as np
import habitat_sim
from habitat_sim.utils.common import d3_40_colors_rgb
import cv2
import json
from time import sleep

def simulate(angle, magnitude, agent_start_pos, furniture_label):
    min_rotate_angle = 0.1
    min_traslate_distance = 0.01
    render_every_frames = 100
    # This is the scene we are going to load.
    # support a variety of mesh formats, such as .glb, .gltf, .obj, .ply
    ### put your scene path ###
    test_scene = "Replica-Dataset/replica_v1/apartment_0/habitat/mesh_semantic.ply"
    path = "Replica-Dataset/replica_v1/apartment_0/habitat/info_semantic.json"

    #global test_pic
    #### instance id to semantic id 
    with open(path, "r") as f:
        annotations = json.load(f)

    id_to_label = []
    instance_id_to_semantic_label_id = np.array(annotations["id_to_label"])
    for i in instance_id_to_semantic_label_id:
        if i < 0:
            id_to_label.append(0)
        else:
            id_to_label.append(i)
    id_to_label = np.asarray(id_to_label)

    ######

    sim_settings = {
        "scene": test_scene,  # Scene path
        "default_agent": 0,  # Index of the default agent
        "sensor_height": 1.5,  # Height of sensors in meters, relative to the agent
        "width": 512,  # Spatial resolution of the observations
        "height": 512,
        "sensor_pitch": 0,  # sensor pitch (x rotation in rads)
    }

    # This function generates a config for the simulator.
    # It contains two parts:
    # one for the simulator backend
    # one for the agent, where you can attach a bunch of sensors

    def transform_rgb_bgr(image):
        return image[:, :, [2, 1, 0]]

    def transform_depth(image):
        depth_img = (image / 10 * 255).astype(np.uint8)
        return depth_img

    def transform_semantic(semantic_obs):
        semantic_img = Image.new("P", (semantic_obs.shape[1], semantic_obs.shape[0]))
        semantic_img.putpalette(d3_40_colors_rgb.flatten())
        # If not our target, label it  0
        semantic_obs_flatten_filter = np.asarray([0 if x != furniture_label else 6 for x in semantic_obs.flatten()])
        semantic_img.putdata(semantic_obs_flatten_filter.astype(np.uint8))

        semantic_img = semantic_img.convert("RGB")

        # Make (120, 120, 120) transparent
        datas = semantic_img.getdata()
        newData = []
        for item in datas:
            if item[0] == 31 and item[1] == 119 and item[2] == 180:
                newData.append((255, 255, 255, 0))
            else:
                newData.append(item)
        semantic_img.putdata(newData)

        semantic_img = cv2.cvtColor(np.asarray(semantic_img), cv2.COLOR_RGB2BGR)
        return semantic_img

    def make_simple_cfg(settings):
        # simulator backend
        sim_cfg = habitat_sim.SimulatorConfiguration()
        sim_cfg.scene_id = settings["scene"]
        

        # In the 1st example, we attach only one sensor,
        # a RGB visual sensor, to the agent
        rgb_sensor_spec = habitat_sim.CameraSensorSpec()
        rgb_sensor_spec.uuid = "color_sensor"
        rgb_sensor_spec.sensor_type = habitat_sim.SensorType.COLOR
        rgb_sensor_spec.resolution = [settings["height"], settings["width"]]
        rgb_sensor_spec.position = [0.0, settings["sensor_height"], 0.0]
        rgb_sensor_spec.orientation = [
            settings["sensor_pitch"],
            0.0,
            0.0,
        ]
        rgb_sensor_spec.sensor_subtype = habitat_sim.SensorSubType.PINHOLE

        #depth snesor
        depth_sensor_spec = habitat_sim.CameraSensorSpec()
        depth_sensor_spec.uuid = "depth_sensor"
        depth_sensor_spec.sensor_type = habitat_sim.SensorType.DEPTH
        depth_sensor_spec.resolution = [settings["height"], settings["width"]]
        depth_sensor_spec.position = [0.0, settings["sensor_height"], 0.0]
        depth_sensor_spec.orientation = [
            settings["sensor_pitch"],
            0.0,
            0.0,
        ]
        depth_sensor_spec.sensor_subtype = habitat_sim.SensorSubType.PINHOLE

        #semantic snesor
        semantic_sensor_spec = habitat_sim.CameraSensorSpec()
        semantic_sensor_spec.uuid = "semantic_sensor"
        semantic_sensor_spec.sensor_type = habitat_sim.SensorType.SEMANTIC
        semantic_sensor_spec.resolution = [settings["height"], settings["width"]]
        semantic_sensor_spec.position = [0.0, settings["sensor_height"], 0.0]
        semantic_sensor_spec.orientation = [
            settings["sensor_pitch"],
            0.0,
            0.0,
        ]
        semantic_sensor_spec.sensor_subtype = habitat_sim.SensorSubType.PINHOLE

        # agent
        agent_cfg = habitat_sim.agent.AgentConfiguration()

        agent_cfg.sensor_specifications = [rgb_sensor_spec, depth_sensor_spec, semantic_sensor_spec]
        ##################################################################
        ### change the move_forward length or rotate angle
        ##################################################################
        agent_cfg.action_space = {
            "move_forward": habitat_sim.agent.ActionSpec(
                "move_forward", habitat_sim.agent.ActuationSpec(amount=min_traslate_distance) # 0.01 means 0.01 m
            ),
            "turn_left": habitat_sim.agent.ActionSpec(
                "turn_left", habitat_sim.agent.ActuationSpec(amount=min_rotate_angle) # 1.0 means 1 degree
            ),
            "turn_right": habitat_sim.agent.ActionSpec(
                "turn_right", habitat_sim.agent.ActuationSpec(amount=min_rotate_angle)
            ),
        }

        return habitat_sim.Configuration(sim_cfg, [agent_cfg])


    cfg = make_simple_cfg(sim_settings)
    sim = habitat_sim.Simulator(cfg)

    # obtain the default, discrete actions that an agent can perform
    # default action space contains 3 actions: move_forward, turn_left, and turn_right
    action_names = list(cfg.agents[sim_settings["default_agent"]].action_space.keys())
    print("Discrete action space: ", action_names)


    # FORWARD_KEY="w"
    # LEFT_KEY="a"
    # RIGHT_KEY="d"
    # FINISH="f"
    # print("#############################")
    # print("use keyboard to control the agent")
    # print(" w for go forward  ")
    # print(" a for turn left  ")
    # print(" d for trun right  ")
    # print(" f for finish and quit the program")
    # print("#############################")

    def navigateAndSee(action=""):
        if action in action_names:
            observations = sim.step(action)
            navigateAndSee.counter = (navigateAndSee.counter + 1) % render_every_frames
            if navigateAndSee.counter == 0:
                img = transform_rgb_bgr(observations["color_sensor"])
                mask = transform_semantic(id_to_label[observations["semantic_sensor"]])
                filter_image = cv2.addWeighted(img, 0.7, mask, 0.3, 0)

                cv2.imshow("RGB", filter_image)
                # navigateAndSee.img_index += 1
                # cv2.imwrite('video_img/' + str(navigateAndSee.img_index) + '.png', filter_image)

                key = cv2.waitKey(1)
                # agent_state = agent.get_state()
                # sensor_state = agent_state.sensor_states['color_sensor']
                # print("camera pose: x y z rw rx ry rz")
                # print(sensor_state.position[0],sensor_state.position[1],sensor_state.position[2],  sensor_state.rotation.w, sensor_state.rotation.x, sensor_state.rotation.y, sensor_state.rotation.z)
    navigateAndSee.img_index = 0
    navigateAndSee.counter = 0

    # initialize an agent
    agent = sim.initialize_agent(sim_settings["default_agent"])

    # Set agent state
    agent_state = habitat_sim.AgentState()
    agent_state.position = np.array(agent_start_pos)  # agent in world space
    agent.set_state(agent_state)

    cur_angle = 180
    img_index = 0
    for i in range(len(angle)):
        # Rotate the agent
        rotate_angle = ((angle[i] - cur_angle) + 180) % 360 - 180
        cur_angle = (cur_angle + rotate_angle) % 360

        if (rotate_angle < 0):
            rotate_angle = abs(rotate_angle)
            action = "turn_right"
            print("{}: {}".format("right", rotate_angle))    
        else:
            action = "turn_left"
            print("{}: {}".format("left", rotate_angle))    
        rotate_times = int(rotate_angle / min_rotate_angle)
        count = 0
        for _ in range(rotate_times):
            navigateAndSee(action)   
  
        # Translate the agnet
        translate_times = int(magnitude[i] / min_traslate_distance)
        count = 0
        for _ in range(translate_times):
            action = "move_forward"
            navigateAndSee(action)

    # action = "move_forward"
    # navigateAndSee(action)

    # while True:
    #     keystroke = cv2.waitKey(0)
    #     if keystroke == ord(FORWARD_KEY):
    #         action = "move_forward"
    #         navigateAndSee(action)
    #         print("action: FORWARD")
    #     elif keystroke == ord(LEFT_KEY):
    #         action = "turn_left"
    #         navigateAndSee(action)
    #         print("action: LEFT")
    #     elif keystroke == ord(RIGHT_KEY):
    #         action = "turn_right"
    #         navigateAndSee(action)
    #         print("action: RIGHT")
    #     elif keystroke == ord(FINISH):
    #         print("action: FINISH")
    #         break
    #     else:
    #         print("INVALID KEY")
    #         continue

