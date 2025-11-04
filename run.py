# Python built-in packages
import importlib.metadata

# Third-party packages
from dotenv import load_dotenv

# Internal modules
from src.main import main

if __name__ == "__main__":
    # Print package name and version, as defined in pyproject.toml
    distribution_metadata = importlib.metadata.metadata("poetrytemplate")
    print(f">> {distribution_metadata['Name']} v.{distribution_metadata['Version']} <<")

    # Load environment variables from local file '.env' (if file exists)
    load_dotenv()

    # Run main function
    main()
