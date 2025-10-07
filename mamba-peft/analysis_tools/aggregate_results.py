from pathlib import Path

def aggregate_specific_text_files(target_dir: str, filename_to_aggregate: str, output_filename: str):
    """
    Scans for subdirectories in the target directory, reads a specific file from each,
    and concatenates them into a single summary file.
    """
    target_dir = Path(target_dir)
    output_file = target_dir / output_filename
    
    if not target_dir.exists():
        print(f"Error: Target directory not found at {target_dir}")
        return

    subdirectories = sorted([d for d in target_dir.iterdir() if d.is_dir()])

    if not subdirectories:
        print(f"No subdirectories found in {target_dir}")
        return

    all_contents = []
    print(f"\nFound {len(subdirectories)} directories to process for '{filename_to_aggregate}'.")

    for exp_dir in subdirectories:
        exp_name = exp_dir.name
        header = f"""
========================================
          Experiment: {exp_name}
========================================
"""
        
        txt_file_path = exp_dir / filename_to_aggregate

        if txt_file_path.exists():
            all_contents.append(header)
            try:
                with open(txt_file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                    all_contents.append(content)
                    print(f"  Successfully read {filename_to_aggregate} from {exp_name}")
            except Exception as e:
                error_msg = f"  ERROR: Could not read file {txt_file_path}. Reason: {e}\n"
                all_contents.append(error_msg)
                print(error_msg)
        else:
            print(f"  INFO: '{filename_to_aggregate}' not found in {exp_name}")

    if not all_contents:
        print(f"No '{filename_to_aggregate}' files were found to aggregate.")
        return

    try:
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write("\n".join(all_contents))
        print(f"\nSuccessfully created summary file at: {output_file}")
    except Exception as e:
        print(f"\nError writing to output file {output_file}: {e}")

def main():
    # This main function is for standalone execution and demonstration.
    # The primary purpose of this script is to be imported as a module.
    import argparse
    parser = argparse.ArgumentParser(description="Aggregate specific text files from subdirectories.")
    parser.add_argument("--target_dir", type=str, required=True, help="Directory containing experiment subfolders.")
    parser.add_argument("--filename", type=str, required=True, help="The exact filename to aggregate from each subfolder.")
    parser.add_argument("--output_filename", type=str, required=True, help="Name for the final aggregated file.")
    args = parser.parse_args()

    aggregate_specific_text_files(
        target_dir=args.target_dir,
        filename_to_aggregate=args.filename,
        output_filename=args.output_filename
    )

if __name__ == "__main__":
    main()