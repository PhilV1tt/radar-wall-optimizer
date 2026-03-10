"""Tests pour le simulateur FDTD 2D TMz."""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pytest
from src.fdtd import FDTD2D_TMz, FDTDConfig, C0, EPS0, MU0


@pytest.fixture
def small_config():
    """Configuration FDTD petite pour tests rapides."""
    return FDTDConfig(
        nx=60, ny=60, ppw=15, freq=10e9, n_steps=100,
        n_pml=8, tfsf_margin=8, wall_center_x=35, wall_center_y=30
    )


class TestFDTDBasics:
    """Tests de base du simulateur FDTD."""

    def test_config_derived_params(self):
        """Vérifie que les paramètres dérivés sont calculés correctement."""
        cfg = FDTDConfig(nx=100, ny=100, ppw=20, freq=10e9, n_pml=10)
        assert cfg.wavelength == pytest.approx(C0 / 10e9)
        assert cfg.dx == pytest.approx(cfg.wavelength / 20)
        assert cfg.dt == pytest.approx(0.5 * cfg.dx / C0)
        assert cfg.nx_total == 120  # 100 + 2*10
        assert cfg.ny_total == 120

    def test_field_allocation(self, small_config):
        """Vérifie les dimensions des champs."""
        sim = FDTD2D_TMz(small_config)
        nxt = small_config.nx_total
        nyt = small_config.ny_total
        assert sim.Ez.shape == (nxt, nyt)
        assert sim.Hx.shape == (nxt, nyt - 1)
        assert sim.Hy.shape == (nxt - 1, nyt)

    def test_initial_fields_zero(self, small_config):
        """Vérifie que les champs sont initialisés à zéro."""
        sim = FDTD2D_TMz(small_config)
        assert np.all(sim.Ez == 0)
        assert np.all(sim.Hx == 0)
        assert np.all(sim.Hy == 0)


class TestFDTDPropagation:
    """Tests de propagation des ondes."""

    def test_source_excitation(self, small_config):
        """Vérifie que la source Ricker excite le champ."""
        sim = FDTD2D_TMz(small_config)
        sim.run(50)
        assert np.max(np.abs(sim.Ez)) > 0

    def test_energy_conservation_free_space(self, small_config):
        """Vérifie que l'énergie dans le domaine est bornée (stabilité)."""
        sim = FDTD2D_TMz(small_config)
        energies = []
        for _ in range(100):
            sim.step()
            energies.append(np.sum(sim.Ez**2))

        max_energy = max(energies)
        assert max_energy < 1e10, "Instabilité détectée : énergie explosive"

    def test_pec_reflection(self, small_config):
        """Vérifie que le PEC réfléchit l'onde (Ez=0 sur conducteur)."""
        sim = FDTD2D_TMz(small_config)
        params = np.zeros(10)
        sim.set_wall_from_params(params, n_segments=10, wall_height=30, wall_thickness=4)
        sim.run(100)

        assert np.all(sim.Ez[sim.pec_mask] == 0)

    def test_physical_fields_extraction(self, small_config):
        """Vérifie que get_physical_fields retourne les bonnes dimensions."""
        sim = FDTD2D_TMz(small_config)
        sim.run(10)
        ez, pec = sim.get_physical_fields()
        assert ez.shape == (small_config.nx, small_config.ny)
        assert pec.shape == (small_config.nx, small_config.ny)


class TestPML:
    """Tests des couches absorbantes PML."""

    def test_pml_coefficients_range(self, small_config):
        """Vérifie que les coefficients PML sont dans des plages valides."""
        sim = FDTD2D_TMz(small_config)
        assert np.all(sim.bx_e >= 0) and np.all(sim.bx_e <= 1)
        assert np.all(sim.by_e >= 0) and np.all(sim.by_e <= 1)
        assert np.all(sim.ax_e <= 0)
        assert np.all(sim.ay_e <= 0)

    def test_pml_interior_is_unity(self, small_config):
        """Vérifie que les coefficients PML sont neutres à l'intérieur."""
        sim = FDTD2D_TMz(small_config)
        n = small_config.n_pml
        assert np.all(sim.bx_e[n:-n] == 1.0)
        assert np.all(sim.ax_e[n:-n] == 0.0)

    def test_pml_absorption(self):
        """Vérifie que la PML absorbe l'onde."""
        cfg = FDTDConfig(
            nx=60, ny=60, ppw=15, freq=10e9, n_steps=50,
            n_pml=10, tfsf_margin=8, wall_center_x=35, wall_center_y=30
        )
        sim = FDTD2D_TMz(cfg)
        n = cfg.n_pml

        sim.run(150)
        energy_mid = np.sum(sim.Ez[n:-n, n:-n]**2)

        sim.run(100)
        energy_late = np.sum(sim.Ez[n:-n, n:-n]**2)

        assert energy_late < energy_mid, "PML ne semble pas absorber"


class TestRAM:
    """Tests des matériaux absorbants radar."""

    def test_ram_reduces_scattering(self):
        """Vérifie que la couche RAM réduit la rétrodiffusion."""
        cfg = FDTDConfig(
            nx=80, ny=80, ppw=15, freq=10e9, n_steps=250,
            n_pml=10, tfsf_margin=8, wall_center_x=50, wall_center_y=40
        )

        sim1 = FDTD2D_TMz(cfg)
        sim1.set_wall_from_params(np.zeros(10), n_segments=10,
                                   wall_height=30, wall_thickness=4)
        sim1.run()
        e_pec = sim1.compute_backscatter_energy()

        sim2 = FDTD2D_TMz(cfg)
        sim2.set_wall_from_params_ram(np.zeros(10), n_segments=10,
                                       wall_height=30, wall_thickness=4,
                                       ram_thickness=2, ram_sigma=1.0)
        sim2.run()
        e_ram = sim2.compute_backscatter_energy()

        assert e_ram < e_pec, "RAM devrait réduire la rétrodiffusion"


class TestReset:
    """Tests de la remise à zéro."""

    def test_reset_clears_fields(self, small_config):
        """Vérifie que reset() remet tout à zéro."""
        sim = FDTD2D_TMz(small_config)
        sim.run(50)
        assert np.max(np.abs(sim.Ez)) > 0

        sim.reset()
        assert np.all(sim.Ez == 0)
        assert np.all(sim.Hx == 0)
        assert np.all(sim.Hy == 0)
        assert sim.time_step == 0
