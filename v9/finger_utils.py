# finger_utils.py

def get_finger_states(landmarks):
    """
    Returns list indicating if each finger is open (1) or closed (0)
    Order: Thumb, Index, Middle, Ring, Pinky
    """
    finger_states = []

    # Thumb (check x-axis instead of y)
    if landmarks[4].x < landmarks[3].x:
        finger_states.append(1)
    else:
        finger_states.append(0)

    # Fingers: tip.y < pip.y means finger is up
    tips_ids = [8, 12, 16, 20]
    pip_ids = [6, 10, 14, 18]

    for tip, pip in zip(tips_ids, pip_ids):
        if landmarks[tip].y < landmarks[pip].y:
            finger_states.append(1)
        else:
            finger_states.append(0)

    return finger_states


def detect_sign(finger_states):
    """
    Map finger states to gestures.
    Order of finger_states: [Thumb, Index, Middle, Ring, Pinky]
    """
    signs = {
        (0, 0, 0, 0, 0): "Fist 👊",
        (1, 1, 1, 1, 1): "Open Palm ✋",
        (1, 0, 0, 0, 0): "Thumbs Up 👍",
        (0, 1, 1, 0, 0): "Victory ✌",
        (0, 1, 0, 0, 0): "Pointing ☝",
        (1, 0, 1, 0, 1): "Rock 🤘",
        (0, 1, 0, 0, 1): "I Love You 🤟",
        (1, 1, 0, 0, 0): "Gun Sign 🔫",
        (1, 1, 1, 0, 0): "Three Fingers",
        (0, 1, 1, 1, 0): "Scout Salute",
        (0, 1, 1, 1, 1): "Four Fingers Up",
        (0, 0, 0, 1, 1): "Ring and Pinky",
        (1, 1, 0, 1, 1): "Spiderman 🕷️",
    }

    return signs.get(tuple(finger_states), "Unknown")
