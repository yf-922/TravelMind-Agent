"""Run the production UI/API on localhost with disposable SQLite storage."""

import argparse
import os
import sys
import tempfile
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--port', type=int, default=8877)
    args = parser.parse_args()
    from app.core import database
    from app.multi_agent_core.memory import SQLiteAgentMemoryStore
    import uvicorn

    with tempfile.TemporaryDirectory(prefix='travelmind-preview-') as directory:
        store = SQLiteAgentMemoryStore(Path(directory) / 'private.db')
        with patch.object(database, '_DB_PATH', Path(directory) / 'application.db'), \
             patch('app.multi_agent_core.memory.SQLiteAgentMemoryStore', return_value=store), \
             patch.dict(os.environ, {'SEMANTIC_MEMORY_ENABLED': '0', 'PLAN_TIMEOUT_SECONDS': '240'}):
            database.init_db()
            import app.main as main_module
            async def no_background(*args, **kwargs):
                return None
            main_module.run_profile_update_agent = no_background
            main_module.run_semantic_memory_update = no_background
            print('Isolated localhost preview: real planning providers; temporary SQLite; semantic/background extraction disabled.', flush=True)
            uvicorn.run(main_module.app, host='127.0.0.1', port=args.port)


if __name__ == '__main__':
    main()
