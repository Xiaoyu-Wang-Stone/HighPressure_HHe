import sys
import ase.units as units
from ase import Atoms
from ase.io import write

# Try to import the i-PI utility module
try:
    from ipi.utils.io import read_file
except ImportError:
    print("Error: i-PI is not installed or not in your PYTHONPATH.")
    sys.exit(1)


def convert_ipi_centroid_to_extxyz(pos_file, output_file):
    """
    Read i-PI centroid coordinates (pos_centroid), convert units,
    and export to Extended XYZ format.
    """
    # Conversion factor: i-PI defaults to atomic units (Bohr)
    # Coordinate conversion: Bohr -> Angstrom
    BOHR_TO_ANGSTROM = units.Bohr

    print(f"Starting conversion...")
    print(f"  Input file: {pos_file}")
    print(f"  Output file: {output_file}")

    frame_count = 0

    # Clear or create output file to avoid duplicate appends
    with open(output_file, "w"):
        pass

    with open(pos_file, "r") as f_pos:
        while True:
            try:
                # Read one frame of coordinates
                ret_pos = read_file("xyz", f_pos)

                # i-PI read_file may raise EOFError at end-of-file,
                # or return an object with empty data depending on version.
                # The main loop termination here relies on EOFError.

                # Build an ASE Atoms object
                # i-PI q is typically flattened and needs reshape
                # i-PI cell matrix h is usually column-vector style;
                # ASE expects row vectors, so transpose with .T
                frame = Atoms(
                    symbols=ret_pos["atoms"].names,
                    positions=ret_pos["atoms"].q.reshape((-1, 3)) * BOHR_TO_ANGSTROM,
                    cell=ret_pos["cell"].h.T * BOHR_TO_ANGSTROM,
                    pbc=True,
                )

                # Write to output file in append mode
                write(output_file, frame, format="extxyz", append=True)

                frame_count += 1
                if frame_count % 100 == 0:
                    print(f"Processed {frame_count} frames...", end="\r")

            except EOFError:
                break
            except Exception as e:
                # Some read_file versions may raise specific errors at EOF,
                # or if input format is invalid (for example blank lines).
                # For empty files or normal EOF, it is usually safe to exit.
                if "end of file" in str(e).lower():
                    break
                print(f"\nError while parsing frame {frame_count + 1}: {e}")
                # For serious parsing errors, stop conversion
                break

    print(f"\nConversion complete. Processed {frame_count} frames.")
    print(f"Output saved to: {output_file}")


if __name__ == "__main__":
    # Input filename (centroid coordinate file only)
    POS_INPUT = "HHe_PIMD_2000K.pos_centroid.xyz"

    # Output filename
    OUTPUT_FILENAME = "HHe_PIMD_2000K_centroid_converted.xyz"

    convert_ipi_centroid_to_extxyz(POS_INPUT, OUTPUT_FILENAME)
