"""Frozen multimodal reference policy and bounded GRPO group queues."""
import argparse
import os
import queue
import threading
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-path', default=os.getenv('MODEL_PATH'), required=not bool(os.getenv('MODEL_PATH')))
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--host', default='127.0.0.1')
    parser.add_argument('--port', type=int, default=59875)
    args = parser.parse_args()
    import torch
    from bottle import Bottle, request, response
    from inference import NaVILAImageInference
    from mobilevla.checkpoints import require_auxiliary_checkpoint
    from mobilevla.grpo import completion_logps, dumps, loads, validate_batch

    inferencer = NaVILAImageInference(args.model_path, device=args.device)
    model = inferencer.model
    require_auxiliary_checkpoint(model)
    model.eval().requires_grad_(False)
    model_id = str(Path(args.model_path).resolve())
    pending, ready = queue.Queue(maxsize=4), queue.Queue(maxsize=4)
    failures = []
    app = Bottle()

    @app.post('/upload')
    def upload():
        try:
            batch = loads(request.body.read())
            validate_batch(batch)
            if batch.get('model_id') != model_id:
                raise ValueError('Reference and generator base model paths differ')
            pending.put_nowait(batch)
            return 'queued'
        except queue.Full:
            response.status = 429
            return 'Queue is full; retry this group'
        except ValueError as exc:
            response.status = 400
            return str(exc)

    @app.get('/get')
    def get():
        if failures:
            response.status = 500
            return failures[0]
        while True:
            try:
                batch = ready.get_nowait()
            except queue.Empty:
                response.status = 204
                return b''
            # Discard groups from previous training runs, or stale policies.
            if batch['run_id'] != request.query.run_id:
                continue
            if batch['policy_version'] != int(request.query.version):
                continue
            response.content_type = 'application/octet-stream'
            return dumps(batch)

    def score():
        try:
            while True:
                batch = pending.get()
                with torch.no_grad():
                    batch['ref_logps'] = completion_logps(model, batch).cpu()
                validate_batch(batch)
                ready.put(batch)
        except Exception as exc:
            failures.append(f'Reference scoring failed: {type(exc).__name__}: {exc}')

    threading.Thread(target=score, daemon=True).start()
    app.run(host=args.host, port=args.port, server='wsgiref')


if __name__ == '__main__':
    main()
