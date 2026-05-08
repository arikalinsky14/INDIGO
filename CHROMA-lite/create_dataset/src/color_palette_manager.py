# color_palette_manager.py
import threading
import math
import colornames as cn  # as requested

class PaletteMatcher:
    """
    Memory-lean matcher that restricts 'closest name' to your own palettes.
    - For full vocabulary: calls cn.find(...) directly.
    - For restricted palettes: uses ΔE00 (CIEDE2000) in CIE Lab (D65) over just those names.

    You provide only *small* palettes (name -> hex). No huge tables kept here.
    """

    _LOCK = threading.RLock()
    _PALETTES = {"__full__": {}}   # name -> {color_name: "#RRGGBB"}
    _ACTIVE = "__full__"
    _LAB_CACHE = {}  # { palette_name: {normalized_name: (L,a,b)} }

    # ---------- Public API ----------
    @classmethod
    def register_palette(cls, name: str, mapping: dict[str, str]) -> None:
        """
        Register a palette by mapping color name -> '#RRGGBB' (small!).
        Names are matched case-insensitively when searching.
        """
        if not name or not isinstance(name, str):
            raise ValueError("Palette name must be a non-empty string.")
        if not mapping:
            raise ValueError("Palette must contain at least one entry.")
        norm = {cls._norm(k): cls._hex_upper(v) for k, v in mapping.items()}
        with cls._LOCK:
            cls._PALETTES[name] = norm
            cls._LAB_CACHE.pop(name, None)  # drop any stale cache for this palette

    @classmethod
    def activate(cls, name: str) -> None:
        """Set the active palette. '__full__' delegates to cn.find(...)."""
        with cls._LOCK:
            if name not in cls._PALETTES:
                raise KeyError(f"Unknown palette: {name}")
            cls._ACTIVE = name

    @classmethod
    def restore(cls) -> None:
        """Convenience: back to full vocabulary (cn.find)."""
        cls.activate("__full__")

    @classmethod
    def active_palette(cls) -> str:
        return cls._ACTIVE

    @classmethod
    def list_palettes(cls) -> list[str]:
        return sorted(cls._PALETTES.keys())

    @classmethod
    def find(cls, *rgb_or_hex) -> str:
        """
        Unified entry point:
        - If active palette is '__full__', return cn.find(...) (library logic).
        - Else, compute the closest among the active palette only using ΔE00.
        Accepts any input cn.find accepts: (r,g,b) or '#RRGGBB' / 'RRGGBB' / '#RGB'.
        """
        pal = cls._ACTIVE
        if pal == "__full__":
            return cn.find(*rgb_or_hex)  # keep library behavior for full vocab

        rgb = cls._to_rgb_tuple(*rgb_or_hex)
        q_lab = cls._rgb_to_lab(rgb)

        best_name = None
        best_d = float("inf")
        mapping = cls._PALETTES[pal]
        cache = cls._LAB_CACHE.setdefault(pal, {})

        for name_norm, hexval in mapping.items():
            lab = cache.get(name_norm)
            if lab is None:
                lab = cls._rgb_to_lab(cls._hex_to_rgb(hexval))
                cache[name_norm] = lab
            d = cls._delta_e_ciede2000(q_lab, lab)  # ΔE00
            if d < best_d:
                best_d, best_name = d, name_norm

        return cls._prettify_name(best_name) if best_name else cn.find(*rgb_or_hex)

    # ---------- Context manager (temporary palette) ----------
    class _UseCtx:
        def __init__(self, outer, palette):
            self._outer = outer
            self._prev = None
            self._palette = palette
        def __enter__(self):
            self._prev = self._outer.active_palette()
            self._outer.activate(self._palette)
        def __exit__(self, exc_type, exc, tb):
            self._outer.activate(self._prev)

    @classmethod
    def use(cls, palette_name: str):
        if palette_name not in cls._PALETTES:
            raise KeyError(f"Unknown palette: {palette_name}")
        return cls._UseCtx(cls, palette_name)

    # ---------- Helpers (parsing, color math) ----------
    @staticmethod
    def _prettify_name(n: str) -> str:
        return " ".join(part.capitalize() for part in n.split())

    @staticmethod
    def _norm(s: str) -> str:
        return " ".join(s.strip().lower().split())

    @staticmethod
    def _hex_upper(h: str) -> str:
        h = h.strip().lstrip("#")
        if len(h) == 3:
            h = "".join(ch*2 for ch in h)
        if len(h) != 6 or any(c not in "0123456789abcdefABCDEF" for c in h):
            raise ValueError(f"Invalid hex color: {h!r}")
        return "#" + h.upper()

    @staticmethod
    def _hex_to_rgb(h: str) -> tuple[int, int, int]:
        h = h.lstrip("#")
        return int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)

    @staticmethod
    def _to_rgb_tuple(*args) -> tuple[int, int, int]:
        if len(args) == 1:
            x = args[0]
            if isinstance(x, str):
                x = x.strip().lstrip("#")
                if len(x) == 3:
                    x = "".join(c*2 for c in x)
                if len(x) != 6:
                    raise ValueError(f"Unrecognized color string: {args[0]!r}")
                return int(x[0:2], 16), int(x[2:4], 16), int(x[4:6], 16)
            if isinstance(x, (tuple, list)) and len(x) >= 3:
                return int(x[0]), int(x[1]), int(x[2])
        elif len(args) >= 3:
            return int(args[0]), int(args[1]), int(args[2])
        raise ValueError(f"Unrecognized inputs: {args!r}")

    # ---- sRGB (D65) -> CIE Lab helpers ----
    @staticmethod
    def _srgb_to_linear(c: float) -> float:
        c = c / 255.0
        return c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4

    @classmethod
    def _rgb_to_xyz(cls, rgb: tuple[int,int,int]) -> tuple[float,float,float]:
        r, g, b = (cls._srgb_to_linear(rgb[0]), cls._srgb_to_linear(rgb[1]), cls._srgb_to_linear(rgb[2]))
        # sRGB -> XYZ (D65) matrix
        X = r*0.4124564 + g*0.3575761 + b*0.1804375
        Y = r*0.2126729 + g*0.7151522 + b*0.0721750
        Z = r*0.0193339 + g*0.1191920 + b*0.9503041
        # Scale to 100
        return (X*100.0, Y*100.0, Z*100.0)

    @staticmethod
    def _f_lab(t: float) -> float:
        # CIE standard: delta = 6/29
        delta = 6/29
        if t > (delta**3):
            return t ** (1/3)
        return (t / (3*delta*delta)) + (4/29)

    @classmethod
    def _xyz_to_lab(cls, xyz: tuple[float,float,float]) -> tuple[float,float,float]:
        # Reference white D65
        Xn, Yn, Zn = 95.047, 100.000, 108.883
        x, y, z = xyz[0]/Xn, xyz[1]/Yn, xyz[2]/Zn
        fx, fy, fz = cls._f_lab(x), cls._f_lab(y), cls._f_lab(z)
        L = 116*fy - 16
        a = 500*(fx - fy)
        b = 200*(fy - fz)
        return (L, a, b)

    @classmethod
    def _rgb_to_lab(cls, rgb: tuple[int,int,int]) -> tuple[float,float,float]:
        return cls._xyz_to_lab(cls._rgb_to_xyz(rgb))

    # ---- CIEDE2000 (ΔE00) ----
    @staticmethod
    def _delta_e_ciede2000(Lab_1, Lab_2, kL=1, kC=1, kH=1):
        """
        Calculate CIEDE2000 color difference between two LAB colors.

        Args:
            Lab_1: First color as (L*, a*, b*) tuple
            Lab_2: Second color as (L*, a*, b*) tuple
            kL, kC, kH: Weighting factors (default 1, 1, 1)

        Returns:
            CIEDE2000 color difference value (ΔE00)
        """

        # Step 1: Calculate C_i, h_i
        L1, a1, b1 = Lab_1[0], Lab_1[1], Lab_1[2]
        L2, a2, b2 = Lab_2[0], Lab_2[1], Lab_2[2]

        C1_ab = math.sqrt(a1 ** 2 + b1 ** 2)
        C2_ab = math.sqrt(a2 ** 2 + b2 ** 2)

        C_ab_bar = (C1_ab + C2_ab) / 2

        C_25_7 = 6103515625  # 25^7

        G = 0.5 * (1 - math.sqrt(C_ab_bar ** 7 / (C_ab_bar ** 7 + C_25_7)))

        a1_prime = (1 + G) * a1
        a2_prime = (1 + G) * a2

        C1_prime = math.sqrt(a1_prime ** 2 + b1 ** 2)
        C2_prime = math.sqrt(a2_prime ** 2 + b2 ** 2)

        # Calculate h_i_prime (hue angles) - CORRECTED
        if b1 == 0 and a1_prime == 0:
            h1_prime = 0
        else:
            h1_prime = math.atan2(b1, a1_prime)
            if h1_prime < 0:
                h1_prime += 2 * math.pi

        if b2 == 0 and a2_prime == 0:
            h2_prime = 0
        else:
            h2_prime = math.atan2(b2, a2_prime)
            if h2_prime < 0:
                h2_prime += 2 * math.pi

        # Step 2: Calculate ΔL', ΔC', ΔH'
        delta_L_prime = L2 - L1
        delta_C_prime = C2_prime - C1_prime

        delta_h_prime = h2_prime - h1_prime
        if C1_prime * C2_prime == 0:
            delta_h_prime = 0
        elif delta_h_prime > math.pi:
            delta_h_prime -= 2 * math.pi
        elif delta_h_prime < -math.pi:
            delta_h_prime += 2 * math.pi

        delta_H_prime = 2 * math.sqrt(C1_prime * C2_prime) * math.sin(delta_h_prime / 2)

        # Step 3: Calculate CIEDE2000 Color-Difference ΔE00
        L_prime_bar = (L1 + L2) / 2
        C_prime_bar = (C1_prime + C2_prime) / 2

        # Calculate h_prime_bar (mean hue)
        abs_diff_h_prime = abs(h1_prime - h2_prime)
        sum_h_prime = h1_prime + h2_prime
        C1C2_prime = C1_prime * C2_prime

        if abs_diff_h_prime <= math.pi and C1C2_prime != 0:
            h_prime_bar = (h1_prime + h2_prime) / 2
        elif abs_diff_h_prime > math.pi and sum_h_prime < 2 * math.pi and C1C2_prime != 0:
            h_prime_bar = (h1_prime + h2_prime) / 2 + math.pi
        elif abs_diff_h_prime > math.pi and sum_h_prime >= 2 * math.pi and C1C2_prime != 0:
            h_prime_bar = (h1_prime + h2_prime) / 2 - math.pi
        else:
            h_prime_bar = h1_prime + h2_prime

        # Calculate T
        T = (1 - 0.17 * math.cos(h_prime_bar - math.pi / 6)
            + 0.24 * math.cos(2 * h_prime_bar)
            + 0.32 * math.cos(3 * h_prime_bar + math.pi / 30)
            - 0.20 * math.cos(4 * h_prime_bar - 63 * math.pi / 180))

        # Calculate Δθ
        h_prime_bar_deg = h_prime_bar * 180 / math.pi
        if h_prime_bar_deg < 0:
            h_prime_bar_deg += 360
        elif h_prime_bar_deg > 360:
            h_prime_bar_deg -= 360

        delta_theta = 30 * math.exp(-(((h_prime_bar_deg - 275) / 25) ** 2))

        # Calculate R_C
        R_C = 2 * math.sqrt(C_prime_bar ** 7 / (C_prime_bar ** 7 + C_25_7))

        # Calculate S_L, S_C, S_H
        L_bar_minus_50_sq = (L_prime_bar - 50) ** 2
        S_L = 1 + (0.015 * L_bar_minus_50_sq) / math.sqrt(20 + L_bar_minus_50_sq)
        S_C = 1 + 0.045 * C_prime_bar
        S_H = 1 + 0.015 * C_prime_bar * T

        # Calculate R_T
        R_T = -math.sin(delta_theta * math.pi / 90) * R_C

        # Calculate ΔE00
        term_L = delta_L_prime / (kL * S_L)
        term_C = delta_C_prime / (kC * S_C)
        term_H = delta_H_prime / (kH * S_H)

        delta_E00 = math.sqrt(
            term_L ** 2
            + term_C ** 2
            + term_H ** 2
            + R_T * term_C * term_H
        )

        return delta_E00

