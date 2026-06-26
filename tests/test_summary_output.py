from phonoflow.reporting.summary_text import format_summary


def test_summary_formats_backend_resolution_and_key_metrics():
    text = format_summary(
        {
            "input_path": "examples/Si.vasp",
            "model_path": "nep.txt",
            "backend_requested": "auto",
            "backend_resolved": "calorine",
            "structure_formula": "Si2",
            "n_atoms_unitcell": 2,
            "supercell_dim_resolved": [2, 2, 2],
            "n_displaced_supercells": 6,
            "relax_converged": True,
            "final_max_force_eV_per_A": 0.005,
            "minimum_frequency_THz": -0.02,
            "maximum_frequency_THz": 15.2,
            "has_imaginary_frequency": False,
            "dos": True,
            "output_files": {"band_plot": "phonon_band.png", "dos_plot": "phonon_dos.png"},
        }
    )

    assert "Backend: auto -> calorine" in text
    assert "Final max force: 0.005 eV/A" in text
    assert "Displaced supercells: 6" in text
