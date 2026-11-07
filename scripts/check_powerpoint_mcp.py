#!/usr/bin/env python3
"""Exercise the local PowerPoint MCP over stdio, without any cloud upload."""
import asyncio
import json
from pathlib import Path
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

ROOT=Path(__file__).resolve().parents[1]
SERVER=Path('/home/sonnguyen112/.local/share/motip-powerpoint-mcp/bin/ppt_mcp_server')


def payload(result):
    assert not result.isError, result
    return json.loads(next(c.text for c in result.content if c.type=='text'))


async def main():
    params=StdioServerParameters(command=str(SERVER),args=[],cwd=str(ROOT))
    async with stdio_client(params) as (read,write):
        async with ClientSession(read,write) as session:
            init=await session.initialize()
            tools=await session.list_tools()
            opened=payload(await session.call_tool('open_presentation',dict(
                file_path=str(ROOT/'presentation/THUYET_TRINH_MOTIP.pptx'),id='course')))
            assert opened['slide_count']==29,opened
            properties=payload(await session.call_tool('set_core_properties',dict(
                presentation_id='course',author='Nguyễn Hồ Trường Sơn — 25C01020',
                title='MOTIP và hình học quỹ đạo đa độ trễ',
                subject='Đồ án Thị giác máy tính — TS. Trương Phước Hưng',
                keywords='MOTIP, DanceTrack, multi-object tracking, multi-lag geometry')))
            assert 'error' not in properties,properties
            saved=payload(await session.call_tool('save_presentation',dict(presentation_id='course',
                file_path=str(ROOT/'presentation/THUYET_TRINH_MOTIP.pptx'))))
            assert 'error' not in saved,saved
            info=payload(await session.call_tool('get_presentation_info',dict(presentation_id='course')))
            assert 'error' not in info,info
            result=dict(server=init.serverInfo.model_dump(),tool_count=len(tools.tools),
                available_tools=[t.name for t in tools.tools],presentation_info=info,
                transport='stdio',upload_performed=False,core_properties_updated=True)
            (ROOT/'sources/POWERPOINT_MCP_CHECK.json').write_text(json.dumps(result,indent=2,ensure_ascii=False)+'\n')
            print(json.dumps(dict(server=result['server'],tools=result['tool_count'],slides=opened['slide_count'],upload_performed=False),ensure_ascii=False))


if __name__=='__main__':asyncio.run(main())
