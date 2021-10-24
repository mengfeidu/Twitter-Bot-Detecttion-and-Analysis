from torch.utils.tensorboard import SummaryWriter

from util import LOG_DIR, Path


def cleanup(path: Path) -> None:
    if path.exists():
        for file in path.iterdir():
            if file.is_dir():
                cleanup(file)
                file.rmdir()
            else:
                file.unlink()


def get_writer(name: str) -> SummaryWriter:
    writer_path: Path = LOG_DIR.joinpath(name)
    cleanup(writer_path)
    return SummaryWriter(writer_path)
