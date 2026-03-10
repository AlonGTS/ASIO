# gts_tracker.pyx
# GTS Tracker - proprietary wrapper
# Build with: python3 setup.py build_ext --inplace

import cv2


# ---------------------------------------------------------------------------
# Internal parameter builder (compiled into binary, not visible to customer)
# ---------------------------------------------------------------------------

def _make_params(moving: bool):
    p = cv2.TrackerCSRT_Params()

    # Common parameters
    _common = dict(
        use_hog             = True,
        use_color_names     = True,
        use_gray            = False,
        use_rgb             = True,
        use_channel_weights = True,
        use_segmentation    = False,
        window_function     = "hann",
        kaiser_alpha        = 3.2,
        hog_clip            = 2.0,
        histogram_bins      = 16,
        background_ratio    = 2,
    )

    # Mode-specific parameters
    _fixed = dict(
        number_of_scales     = 33,
        scale_step           = 1.03,
        scale_lr             = 0.15,
        scale_sigma_factor   = 0.25,
        scale_model_max_area = 512,
        admm_iterations      = 9,
        template_size        = 160,
        filter_lr            = 0.08,
    )

    _moving = dict(
        number_of_scales     = 55,
        scale_step           = 1.02,
        scale_lr             = 0.65,
        scale_sigma_factor   = 0.30,
        scale_model_max_area = 1024,
        admm_iterations      = 6,
        template_size        = 200,
        filter_lr            = 0.25,
    )

    for name, val in {**_common, **(_moving if moving else _fixed)}.items():
        if hasattr(p, name):
            setattr(p, name, val)

    return p


def _create_tracker(moving: bool):
    try:
        return cv2.TrackerCSRT_create(_make_params(moving))
    except TypeError:
        t = cv2.TrackerCSRT_create()
        p = _make_params(moving)
        for k, v in p.__dict__.items():
            try:
                getattr(t, k)
                setattr(t, k, v)
            except Exception:
                pass
        return t


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

class GTSTracker:
    """
    GTS proprietary tracker.
    mode: "fixed"  - optimised for stationary targets
          "moving" - optimised for moving targets
    """

    def __init__(self, mode="fixed"):
        if mode not in ("fixed", "moving"):
            raise ValueError("mode must be 'fixed' or 'moving'")
        self._moving = (mode == "moving")
        self._tracker = None
        self._initialised = False

    def init(self, frame, bbox):
        """
        Initialise the tracker on frame at bbox (x, y, w, h).
        Must be called before update().
        """
        self._tracker = _create_tracker(self._moving)
        self._tracker.init(frame, bbox)
        self._initialised = True

    def update(self, frame):
        """
        Update tracker with next frame.
        Returns (success: bool, bbox: tuple(x, y, w, h))
        Returns (False, None) if not initialised.
        """
        if not self._initialised or self._tracker is None:
            return False, None
        success, bbox = self._tracker.update(frame)
        return success, tuple(map(int, bbox)) if success else None

    def reset(self):
        """Clear tracker state. Call init() again to restart."""
        self._tracker = None
        self._initialised = False

    @property
    def mode(self):
        return "moving" if self._moving else "fixed"
