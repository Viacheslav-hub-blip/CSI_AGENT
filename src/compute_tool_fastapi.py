#!/usr/bin/env python
# coding: utf-8

# In[79]:

JOB_QUEUES: dict[str, asyncio.Queue] = {}
JOB_STATUS: dict[str, str] = {}

# # Сервер

# In[83]:

async def process_dataframe(df: pd.DataFrame, user_input: str, q) -> pd.DataFrame:
    '''Преобразование над Dataframe'''
    logger.info('##SERVER_PROCESS_DATAFRAME##', extra = {'extra_info': f''})
    agent = ProcessDataFrameAgent(df, GigaChat_Max)
    async for chunk in agent.run_with_streaming(
        {"user_input": user_input}
    ):
        await q.put(await sse_event(json.dumps({"type": "info", "msg": chunk}), event="info"))
        await asyncio.sleep(0.15)
    return final_df

async def sse_event(data: str, event: str = None) -> str:
    '''Формирование SSE события'''
    lines = []
    if event:
        lines.append(f"event: {event}")
    for chunk in data.splitlines():
        lines.append(f"data: {chunk}")
    lines.append("")
    logger.info('##SERVER_SSE_EVENT##', extra = {'extra_info': f'сформированное событие: {" ".join(lines) + ": keep-alive"}'})
    return "\n".join(lines) + "\n"

async def worker_compute(job_id: str, dataframe_path: Path, user_input: str):
    '''Процесс, который загружает файл, запускает обработку и сохрнаяет результат'''
    logger.info('##WORKER_COMPUTE##', extra = {'extra_info': f'Переданные параметры: {job_id}, {user_input}'})
    q = JOB_QUEUES[job_id]
    loop = asyncio.get_running_loop()
    try:
        logger.info('##WORKER_COMPUTE##', extra = {'extra_info': f'JOB_QUEUES {q}'})
        JOB_STATUS[job_id] = "running"
        await q.put(await sse_event(json.dumps({"type": "info", "msg": "loading imput pickle"}), event="info"))
        await asyncio.sleep(0.15)
        df = await loop.run_in_executor(None, pd.read_pickle, str(dataframe_path))
        if len(df) > 500:
            raise ValueError("To large dataframe")
        df = await process_dataframe(df, user_input, q)
        result_path = RESULT_DIR / f"{job_id}.pkl"
        await q.put(await sse_event(json.dumps({"type": "info", "msg": "saving result"}), event="info"))
        await asyncio.sleep(0.15)
        await loop.run_in_executor(None, df.to_pickle, str(result_path))
        result_url = f"/results/{job_id}/result.pkl"
        payload = {"type": "result", "job_id": job_id, "result_url": result_url}
        await q.put(await sse_event(json.dumps(payload), event="result"))
        await asyncio.sleep(0.15)
        JOB_STATUS[job_id] = "done"
        logger.info('##WORKER_COMPUTE## JOB_STATUS', extra = {'extra_info': f'JOB_STATUS done'})
    except Exception as e:
        JOB_STATUS[job_id] = "error"
        logger.info('##WORKER_COMPUTE## JOB_STATUS', extra = {'extra_info': f'JOB_STATUS error'})
        await q.put(await sse_event(json.dumps({"type": "error", "error": str(e)}), event="error"))
        await asyncio.sleep(0.15)
    finally:
        await q.put("__end__")
        await asyncio.sleep(0.15)
        del JOB_QUEUES[job_id]
        try:
            if input_path.exists():
                input_path.unlink()
        except Exception:
            pass

@app.post("/jobs")
async def create_job(file: UploadFile = File(...), user_input: str = Form(...)):
    ''' Принимает
    - pickle файл с dataframe
    - user_input - запрос пользователя для обработки данных
    Возвращает: job_id и endpoint для SSE
    '''
    logger.info('##CREATE_JOB##', extra = {'extra_info': f'Параметры: {user_input}'})
    job_id = str(uuid.uuid4())
    save_path = UPLOAD_DIR / f"{job_id}_input.pkl"
    try:
        async with aiofiles.open(save_path, 'wb') as out_f:
            while True:
                chunk = await file.read(1024*64)
                if not chunk:
                    break
                await out_f.write(chunk)
        logger.info('##CREATE_JOB## SAVED FILE', extra = {'extra_info': f'success'})
    except Exception as e:
        logger.info('##CREATE_JOB## ERROR', extra = {'extra_info': f'failed to save upload file {e}'})
        raise HTTPException(status_code=500, detail=f"failed to save upload file {e}")
    q: asyncio.Queue = asyncio.Queue()
    JOB_QUEUES[job_id] = q
    JOB_STATUS[job_id] = "queued"
    asyncio.create_task(worker_compute(job_id, save_path, user_input))
    return {
        "job_id": job_id,
        "sse_url": f"/events/{job_id}",
        "result_hint": f"/results/{job_id}/result.pkl"
    }

@app.get("/events/{job_id}")
async def events(job_id: str):
    '''Отправляет состояние работы сервиса'''
    logger.info('##EVENTS##', extra = {'extra_info': f'JOB_ID: {job_id}'})
    if job_id not in JOB_QUEUES:
        logger.info('##EVENTS## ERROR', extra = {'extra_info': f'job not found JOB_ID: {job_id}'})
        raise HTTPException(status_code=404, detail="job not found")
    q = JOB_QUEUES[job_id]
    async def event_generator():
        yield await sse_event(json.dumps({"type": "connected", "job_id": job_id}), event="connected")
        heartbeat_interval = 15
        last_hb = time.time()
        while True:
            try:
                ev = await asyncio.wait_for(q.get(), timeout=heartbeat_interval)
            except:
                yield ": keep-alive\n\n"
                continue
            if ev == "__end__":
                yield await sse_event(json.dumps({"type": "closed"}), event="closed")
                break
            yield ev
    return StreamingResponse(event_generator(), media_type="text/event-stream", headers={"Cache-Control": "no-cache", "X-Accel-Buffering":"no"})

@app.get("/results/{job_id}/result.pkl")
async def download_result(job_id: str):
    ''' Возвращает файл c результатом (обновленным dataframe) '''
    logger.info('##DOWNLOAD_RESULT##', extra = {'extra_info': f'JOB_ID: {job_id}'})
    path = RESULT_DIR / f"{job_id}.pkl"
    if not path.exists():
        logger.info('##DOWNLOAD_RESULT## error', extra = {'extra_info': f'result not found JOB_ID: {job_id}'})
        raise HTTPException(status_code=404, detail="result not found")
    return FileResponse(path, media_type="application/octet-straem", filename=f"{job_id}_result.pkl")

# In[ ]:

get_ipython().system('jupyter nbconvert --to python ComputeTool_FastApi.ipynb')