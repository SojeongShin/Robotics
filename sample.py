import mujoco
from mujoco import viewer
import time

# XML 모델 정의 (간단한 공 하나가 중력에 의해 떨어지는 모델)
model = mujoco.MjModel.from_xml_string("""
<mujoco>
    <option gravity="0 0 -9.81"/>
    <worldbody>
        <body name="ball" pos="0 0 1">
            <geom type="sphere" size="0.1" rgba="1 0 0 0.5"/>
            <joint type="free"/>
        </body>
        <geom type="plane" size="5 5 0.1" rgba="0.2 0.2 0.2 1"/>
    </worldbody>
</mujoco>
""")

data = mujoco.MjData(model)

# MuJoCo 뷰어 띄우기
with viewer.launch_passive(model, data) as v:
    start = time.time()
    while v.is_running():
        mujoco.mj_step(model, data)
        v.sync()
        time.sleep(0.01)  # 약간의 delay로 CPU 과부하 방지
