import os
import time
import urllib.request
import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python as mpTasks
from mediapipe.tasks.python import vision

baseDir = os.path.dirname(os.path.abspath(__file__))
memeDir = os.path.join(baseDir, "memes")
memeExtensions = [".jpeg", ".jpg", ".png"]

handModelPath = os.path.join(baseDir, "hand_landmarker.task")
faceModelPath = os.path.join(baseDir, "face_landmarker.task")
handModelUrl = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
faceModelUrl = "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task"

cameraIndex = 0
memeSide = "right"
memeWidthRatio = 0.25
memeHeightRatio = 0.7
memeMargin = 20
mirrorView = True
showDebug = False

thresholds = {
    "shhDistance": 0.28,
    "biteDistance": 0.22,
    "cheekDistance": 0.45,
    "earDistance": 0.40,
    "prayDistance": 1.10,
    "tongueOut": 0.30,
    "laughJaw": 0.45,
    "laughSmile": 0.35,
    "squintEyes": 0.50,
    "squintPucker": 0.35,
    "wideEyes": 0.45,
    "lookUp": 0.45,
    "frown": 0.40,
    "worried": 0.60,
}

debugKeys = [
    "tongueOut", "jawOpen", "mouthSmileLeft", "eyeWideLeft", "eyeLookUpLeft",
    "mouthFrownLeft", "eyeSquintLeft", "mouthPucker", "browInnerUp",
]

handConnections = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (17, 18), (18, 19), (19, 20),
    (0, 17),
]


def ensureModel(path, url):
    if os.path.exists(path):
        return
    print(f"mengunduh model {os.path.basename(path)} (hanya sekali)...")
    try:
        urllib.request.urlretrieve(url, path)
    except Exception as error:
        raise SystemExit(
            f"gagal mengunduh model: {error}\n"
            f"unduh manual dari:\n{url}\nlalu taruh di:\n{path}"
        )


def loadMemeImage(name):
    for extension in memeExtensions:
        path = os.path.join(memeDir, name + extension)
        if os.path.exists(path):
            img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
            if img is not None:
                return img
    print(f"[peringatan] meme '{name}' tidak ditemukan di folder memes, pakai placeholder")
    img = np.full((300, 300, 3), 60, dtype=np.uint8)
    cv2.putText(img, name.upper(), (20, 160), cv2.FONT_HERSHEY_SIMPLEX,
                1.0, (255, 255, 255), 2, cv2.LINE_AA)
    return img


def resizeToFit(img, maxWidth, maxHeight):
    h, w = img.shape[:2]
    scale = min(maxWidth / w, maxHeight / h)
    newW = max(1, int(w * scale))
    newH = max(1, int(h * scale))
    return cv2.resize(img, (newW, newH), interpolation=cv2.INTER_AREA)


def overlayImage(frame, overlay, posX, posY):
    frameH, frameW = frame.shape[:2]
    overH, overW = overlay.shape[:2]

    x1 = max(posX, 0)
    y1 = max(posY, 0)
    x2 = min(posX + overW, frameW)
    y2 = min(posY + overH, frameH)
    if x1 >= x2 or y1 >= y2:
        return frame

    overCropped = overlay[y1 - posY:y2 - posY, x1 - posX:x2 - posX]
    roi = frame[y1:y2, x1:x2]

    if overCropped.ndim == 2:
        overCropped = cv2.cvtColor(overCropped, cv2.COLOR_GRAY2BGR)
    if overCropped.shape[2] == 4:
        colorPart = overCropped[:, :, :3].astype(np.float32)
        alpha = overCropped[:, :, 3:4].astype(np.float32) / 255.0
    else:
        colorPart = overCropped.astype(np.float32)
        alpha = np.ones((*overCropped.shape[:2], 1), dtype=np.float32)

    blended = colorPart * alpha + roi.astype(np.float32) * (1.0 - alpha)
    frame[y1:y2, x1:x2] = blended.astype(np.uint8)
    return frame


def toPixel(landmark, frameW, frameH):
    return np.array([landmark.x * frameW, landmark.y * frameH], dtype=np.float32)


def dist(a, b):
    return float(np.linalg.norm(a - b))


def getFingerStates(hand, frameW, frameH):
    wrist = toPixel(hand[0], frameW, frameH)
    fingerIds = {"index": (8, 6), "middle": (12, 10), "ring": (16, 14), "pinky": (20, 18)}
    states = {}
    for name, (tipId, pipId) in fingerIds.items():
        tip = toPixel(hand[tipId], frameW, frameH)
        pip = toPixel(hand[pipId], frameW, frameH)
        states[name] = dist(tip, wrist) > dist(pip, wrist)
    return states


def buildContext(hands, face, blend, frameW, frameH):
    ctx = {
        "w": frameW,
        "h": frameH,
        "hands": hands,
        "face": face,
        "blend": blend,
        "states": [getFingerStates(hand, frameW, frameH) for hand in hands],
        "palms": [toPixel(hand[9], frameW, frameH) for hand in hands],
        "handSizes": [dist(toPixel(hand[0], frameW, frameH), toPixel(hand[9], frameW, frameH))
                      for hand in hands],
        "faceWidth": None,
    }
    if face is not None:
        upperLip = toPixel(face[13], frameW, frameH)
        lowerLip = toPixel(face[14], frameW, frameH)
        faceLeft = toPixel(face[234], frameW, frameH)
        faceRight = toPixel(face[454], frameW, frameH)
        eyeLeft = toPixel(face[33], frameW, frameH)
        eyeRight = toPixel(face[263], frameW, frameH)
        ctx["mouth"] = (upperLip + lowerLip) / 2.0
        ctx["faceWidth"] = dist(faceLeft, faceRight)
        ctx["eyeLineY"] = float((eyeLeft[1] + eyeRight[1]) / 2.0)
        ctx["leftSide"] = [faceLeft, toPixel(face[50], frameW, frameH)]
        ctx["rightSide"] = [faceRight, toPixel(face[280], frameW, frameH)]
    return ctx


def blendValue(ctx, name):
    return ctx["blend"].get(name, 0.0)


def blendAverage(ctx, leftName, rightName):
    return (blendValue(ctx, leftName) + blendValue(ctx, rightName)) / 2.0


def nearPoints(point, points, limit):
    return any(dist(point, p) < limit for p in points)


def onlyIndexUp(states):
    return (states["index"] and not states["middle"]
            and not states["ring"] and not states["pinky"])


def checkPray(ctx):
    if len(ctx["hands"]) < 2:
        return False
    size = (ctx["handSizes"][0] + ctx["handSizes"][1]) / 2.0
    limit = size * thresholds["prayDistance"]
    palmsClose = dist(ctx["palms"][0], ctx["palms"][1]) < limit
    tipA = toPixel(ctx["hands"][0][12], ctx["w"], ctx["h"])
    tipB = toPixel(ctx["hands"][1][12], ctx["w"], ctx["h"])
    return palmsClose and dist(tipA, tipB) < limit


def checkHeadGrab(ctx):
    if ctx["face"] is None or len(ctx["hands"]) < 2:
        return False
    above = [palm[1] < ctx["eyeLineY"] for palm in ctx["palms"]]
    return sum(above) >= 2


def checkCheeks(ctx):
    if ctx["face"] is None or len(ctx["hands"]) < 2:
        return False
    limit = ctx["faceWidth"] * thresholds["cheekDistance"]
    palms = ctx["palms"]
    for i in range(len(palms)):
        for j in range(len(palms)):
            if i == j:
                continue
            if (nearPoints(palms[i], ctx["leftSide"], limit)
                    and nearPoints(palms[j], ctx["rightSide"], limit)):
                return True
    return False


def checkShh(ctx):
    if ctx["face"] is None:
        return False
    limit = ctx["faceWidth"] * thresholds["shhDistance"]
    for hand, states in zip(ctx["hands"], ctx["states"]):
        if onlyIndexUp(states):
            tip = toPixel(hand[8], ctx["w"], ctx["h"])
            if dist(tip, ctx["mouth"]) < limit:
                return True
    return False


def checkPeace(ctx):
    for states in ctx["states"]:
        if (states["index"] and states["middle"]
                and not states["ring"] and not states["pinky"]):
            return True
    return False


def checkRock(ctx):
    for states in ctx["states"]:
        if (states["index"] and states["pinky"]
                and not states["middle"] and not states["ring"]):
            return True
    return False


def checkBite(ctx):
    if ctx["face"] is None:
        return False
    limit = ctx["faceWidth"] * thresholds["biteDistance"]
    for hand, states in zip(ctx["hands"], ctx["states"]):
        if states["index"]:
            continue
        for tipId in (4, 8):
            tip = toPixel(hand[tipId], ctx["w"], ctx["h"])
            if dist(tip, ctx["mouth"]) < limit:
                return True
    return False


def checkEar(ctx):
    if ctx["face"] is None:
        return False
    limit = ctx["faceWidth"] * thresholds["earDistance"]
    sidePoints = ctx["leftSide"] + ctx["rightSide"]
    return any(nearPoints(palm, sidePoints, limit) for palm in ctx["palms"])


def checkPalm(ctx):
    for states in ctx["states"]:
        if all(states.values()):
            return True
    return False


def checkTongue(ctx):
    return blendValue(ctx, "tongueOut") > thresholds["tongueOut"]


def checkLaugh(ctx):
    return (blendValue(ctx, "jawOpen") > thresholds["laughJaw"]
            and blendAverage(ctx, "mouthSmileLeft", "mouthSmileRight") > thresholds["laughSmile"])


def checkSquint(ctx):
    return (blendAverage(ctx, "eyeSquintLeft", "eyeSquintRight") > thresholds["squintEyes"]
            and blendValue(ctx, "mouthPucker") > thresholds["squintPucker"])


def checkWideEyes(ctx):
    return blendAverage(ctx, "eyeWideLeft", "eyeWideRight") > thresholds["wideEyes"]


def checkLookUp(ctx):
    return blendAverage(ctx, "eyeLookUpLeft", "eyeLookUpRight") > thresholds["lookUp"]


def checkFrown(ctx):
    return blendAverage(ctx, "mouthFrownLeft", "mouthFrownRight") > thresholds["frown"]


def checkWorried(ctx):
    return blendValue(ctx, "browInnerUp") > thresholds["worried"]


memeConfigs = [
    {"name": "pray", "check": checkPray},
    {"name": "headgrab", "check": checkHeadGrab},
    {"name": "cheeks", "check": checkCheeks},
    {"name": "shh", "check": checkShh},
    {"name": "peace", "check": checkPeace},
    {"name": "rock", "check": checkRock},
    {"name": "bite", "check": checkBite},
    {"name": "ear", "check": checkEar},
    {"name": "palm", "check": checkPalm},
    {"name": "tongue", "check": checkTongue},
    {"name": "laugh", "check": checkLaugh},
    {"name": "squint", "check": checkSquint},
    {"name": "wideeyes", "check": checkWideEyes},
    {"name": "lookup", "check": checkLookUp},
    {"name": "frown", "check": checkFrown},
    {"name": "worried", "check": checkWorried},
]


def findActiveMeme(ctx):
    for config in memeConfigs:
        if config["check"](ctx):
            return config["name"]
    return None


def drawHand(frame, hand, frameW, frameH):
    points = [(int(p.x * frameW), int(p.y * frameH)) for p in hand]
    for start, end in handConnections:
        cv2.line(frame, points[start], points[end], (0, 255, 0), 2, cv2.LINE_AA)
    for point in points:
        cv2.circle(frame, point, 4, (0, 0, 255), -1, cv2.LINE_AA)


def drawDebug(frame, ctx, activeMeme):
    lines = [f"meme: {activeMeme}"]
    lines += [f"{key}: {blendValue(ctx, key):.2f}" for key in debugKeys]
    for index, states in enumerate(ctx["states"]):
        lines.append(f"tangan {index}: " + " ".join(k[0] + ("1" if v else "0") for k, v in states.items()))
    for row, text in enumerate(lines):
        cv2.putText(frame, text, (10, 60 + row * 22), cv2.FONT_HERSHEY_SIMPLEX,
                    0.55, (0, 255, 255), 1, cv2.LINE_AA)


def drawMeme(frame, memeImg):
    frameH, frameW = frame.shape[:2]
    memeH, memeW = memeImg.shape[:2]
    if memeSide == "left":
        posX = memeMargin
    else:
        posX = frameW - memeW - memeMargin
    posY = (frameH - memeH) // 2
    return overlayImage(frame, memeImg, posX, posY)


def main():
    ensureModel(handModelPath, handModelUrl)
    ensureModel(faceModelPath, faceModelUrl)

    memeImages = {config["name"]: loadMemeImage(config["name"]) for config in memeConfigs}
    resizedCache = {}

    cap = cv2.VideoCapture(cameraIndex)
    if not cap.isOpened():
        print("webcam tidak bisa dibuka. cek izin kamera atau ubah cameraIndex.")
        return

    handOptions = vision.HandLandmarkerOptions(
        base_options=mpTasks.BaseOptions(model_asset_path=handModelPath),
        running_mode=vision.RunningMode.VIDEO,
        num_hands=2,
        min_hand_detection_confidence=0.6,
        min_hand_presence_confidence=0.6,
        min_tracking_confidence=0.6,
    )
    faceOptions = vision.FaceLandmarkerOptions(
        base_options=mpTasks.BaseOptions(model_asset_path=faceModelPath),
        running_mode=vision.RunningMode.VIDEO,
        num_faces=1,
        output_face_blendshapes=True,
        min_face_detection_confidence=0.6,
        min_face_presence_confidence=0.6,
        min_tracking_confidence=0.6,
    )

    lastTimestamp = 0
    debugOn = showDebug

    with vision.HandLandmarker.create_from_options(handOptions) as handLandmarker, \
         vision.FaceLandmarker.create_from_options(faceOptions) as faceLandmarker:

        while True:
            ok, frame = cap.read()
            if not ok:
                break

            if mirrorView:
                frame = cv2.flip(frame, 1)
            frameH, frameW = frame.shape[:2]

            rgb = np.ascontiguousarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            mpImage = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

            timestamp = int(time.monotonic() * 1000)
            if timestamp <= lastTimestamp:
                timestamp = lastTimestamp + 1
            lastTimestamp = timestamp

            handResult = handLandmarker.detect_for_video(mpImage, timestamp)
            faceResult = faceLandmarker.detect_for_video(mpImage, timestamp)

            hands = handResult.hand_landmarks or []
            face = faceResult.face_landmarks[0] if faceResult.face_landmarks else None
            blend = {}
            if faceResult.face_blendshapes:
                blend = {c.category_name: c.score for c in faceResult.face_blendshapes[0]}

            for hand in hands:
                drawHand(frame, hand, frameW, frameH)

            ctx = buildContext(hands, face, blend, frameW, frameH)
            activeMeme = findActiveMeme(ctx)

            if activeMeme is not None:
                maxW = int(frameW * memeWidthRatio)
                maxH = int(frameH * memeHeightRatio)
                cacheKey = (activeMeme, maxW, maxH)
                if cacheKey not in resizedCache:
                    resizedCache[cacheKey] = resizeToFit(memeImages[activeMeme], maxW, maxH)
                frame = drawMeme(frame, resizedCache[cacheKey])
                cv2.putText(frame, activeMeme.upper(), (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                            0.9, (0, 255, 0), 2, cv2.LINE_AA)

            if debugOn:
                drawDebug(frame, ctx, activeMeme)

            cv2.imshow("gesture meme", frame)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
            if key == ord("d"):
                debugOn = not debugOn

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
