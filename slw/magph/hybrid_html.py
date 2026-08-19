"""Standalone interactive HTML export for magnon-phonon hybrid bands."""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np

MEV_PER_THZ = 4.135667696


def _round_list(arr, ndigits=7):
    return np.round(np.asarray(arr, dtype=np.float64), int(ndigits)).tolist()


def _object_scalar(x):
    arr = np.asarray(x)
    if arr.shape == ():
        return str(arr.item())
    return str(x)


def hybrid_to_html_payload(data, *, kdist=None, tick_indices=None, kpath_symbols=None, source=None) -> dict:
    """Convert a HybridHamiltonian-like object to a compact JSON payload."""
    hybrid = np.asarray(data.eigenvalues[0], dtype=np.float64)
    magnon = np.asarray(data.magnon_energies[0], dtype=np.float64)
    phonon = np.asarray(data.phonon_energies, dtype=np.float64)
    weights = np.asarray(data.magnon_weight[0], dtype=np.float64)
    chirality = np.asarray(data.hybrid_chirality[0], dtype=np.float64)
    qpts = np.asarray(data.qpts_frac, dtype=np.float64)
    if kdist is None:
        kdist_arr = np.arange(qpts.shape[0], dtype=np.float64)
    else:
        kdist_arr = np.asarray(kdist, dtype=np.float64)
    if tick_indices is None:
        tick_arr = np.asarray([0, qpts.shape[0] - 1], dtype=np.int32)
    else:
        tick_arr = np.asarray(tick_indices, dtype=np.int32)
    if kpath_symbols is None:
        labels = [str(int(x)) for x in tick_arr.tolist()]
    else:
        labels = [
            str(x).replace("$\\Gamma$", "Γ").replace("\\Gamma", "Γ").replace("$", "")
            for x in np.asarray(kpath_symbols, dtype=object).tolist()
        ]

    return {
        "source": "" if source is None else str(source),
        "component": _object_scalar(data.component),
        "units": _object_scalar(getattr(data, "units", "meV")),
        "mevPerThz": MEV_PER_THZ,
        "kdist": _round_list(kdist_arr, 8),
        "qpts": _round_list(qpts, 6),
        "hybrid": _round_list(hybrid, 6),
        "magnonWeight": _round_list(weights, 6),
        "hybridChirality": _round_list(chirality, 6),
        "bareMagnon": _round_list(magnon, 6),
        "barePhonon": _round_list(phonon, 6),
        "tickIndices": tick_arr.tolist(),
        "tickLabels": labels,
        "stats": {
            "nq": int(hybrid.shape[0]),
            "nHybrid": int(hybrid.shape[1]),
            "nMagnon": int(magnon.shape[1]),
            "nPhonon": int(phonon.shape[1]),
            "emin": float(np.nanmin(hybrid)),
            "emax": float(np.nanmax(hybrid)),
            "maxVertexMev": float(np.nanmax(np.abs(np.asarray(data.magnon_vertex)))),
            "minPhonon": float(np.nanmin(phonon)),
            "minMagnon": float(np.nanmin(magnon)),
        },
    }


def save_hybrid_html_data(path, data, *, kdist=None, tick_indices=None, kpath_symbols=None, source=None) -> None:
    payload = hybrid_to_html_payload(
        data,
        kdist=kdist,
        tick_indices=tick_indices,
        kpath_symbols=kpath_symbols,
        source=source,
    )
    Path(path).write_text(json.dumps(payload, separators=(",", ":")), encoding="utf-8")


def _html_template(data_url: str) -> str:
    data_url_json = json.dumps(str(data_url))
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"/><meta name="viewport" content="width=device-width,initial-scale=1"/>
<title>Hybrid magnon-phonon bands</title>
<style>
:root{{--bg:#f6f7f9;--panel:#fff;--ink:#15171a;--muted:#5b6472;--grid:#d8dde6;--magnon:#b91c1c;--phonon:#8a94a6}}
*{{box-sizing:border-box}}body{{margin:0;min-height:100vh;background:var(--bg);color:var(--ink);font-family:Inter,ui-sans-serif,system-ui,-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif}}
main{{width:min(1500px,100vw);margin:0 auto;padding:clamp(14px,2vw,28px)}}.slide{{aspect-ratio:16/9;min-height:560px;background:var(--panel);border:1px solid #e2e6ee;border-radius:8px;box-shadow:0 18px 50px rgba(22,34,51,.12);display:grid;grid-template-rows:auto auto 1fr auto;gap:12px;padding:clamp(18px,2.2vw,34px)}}
header{{display:grid;grid-template-columns:minmax(0,1fr) auto;gap:18px;align-items:start}}h1{{margin:0;font-size:clamp(26px,2.6vw,46px);line-height:1.05;font-weight:760;letter-spacing:0}}.subtitle{{margin-top:8px;color:var(--muted);font-size:clamp(13px,1.1vw,17px)}}
.stats{{display:grid;grid-template-columns:repeat(4,minmax(110px,1fr));gap:8px}}.stat{{border:1px solid #e3e7ef;border-radius:8px;padding:8px 10px;min-width:0}}.stat .label{{color:var(--muted);font-size:11px;text-transform:uppercase;letter-spacing:.04em}}.stat .value{{margin-top:3px;font-size:clamp(13px,1.2vw,18px);font-weight:720;white-space:nowrap}}
.controls{{display:flex;flex-wrap:wrap;gap:10px 18px;align-items:center;color:var(--muted);font-size:13px;min-height:30px}}.controls label{{display:inline-flex;gap:7px;align-items:center}}.controls select,.controls input[type=number],.controls button{{height:30px;border:1px solid #cdd4df;border-radius:6px;background:#fff;color:var(--ink);padding:0 8px;font:inherit}}.controls button{{padding:0 10px;cursor:pointer}}.controls input[type=file]{{max-width:210px}}.controls input[type=checkbox]{{width:16px;height:16px}}
.plot-wrap{{position:relative;min-height:0;border-top:1px solid #edf0f5;padding-top:8px}}canvas{{width:100%;height:100%;display:block}}.tooltip{{position:absolute;pointer-events:none;background:rgba(255,255,255,.96);border:1px solid #d8dee9;border-radius:8px;padding:8px 10px;color:var(--ink);font-size:12px;box-shadow:0 10px 28px rgba(22,34,51,.16);transform:translate(12px,12px);display:none;min-width:190px}}
.footer{{display:flex;justify-content:space-between;align-items:center;gap:16px;color:var(--muted);font-size:12px}}.legend{{display:inline-flex;gap:16px;align-items:center;flex-wrap:wrap}}.swatch{{display:inline-flex;gap:6px;align-items:center}}.line{{width:24px;height:0;border-top:3px solid #1464f4;display:inline-block}}.line.ph{{border-color:var(--phonon);border-top-width:2px}}.line.mag{{border-color:var(--magnon);border-top-style:dashed}}.status{{color:#9a3412}}
@media(max-width:900px){{.slide{{aspect-ratio:auto;min-height:760px}}header{{grid-template-columns:1fr}}.stats{{grid-template-columns:repeat(2,minmax(0,1fr))}}}}
</style></head>
<body><main><section class="slide"><header><div><h1>Hybrid magnon-phonon bands</h1><div class="subtitle">Hybridization along <strong id="pathLabel">loading</strong></div></div><div class="stats"><div class="stat"><div class="label">Component</div><div class="value" id="component">-</div></div><div class="stat"><div class="label">Hybrid modes</div><div class="value" id="modeCount">-</div></div><div class="stat"><div class="label">Energy range</div><div class="value" id="energyRange">-</div></div><div class="stat"><div class="label">Max |g|</div><div class="value" id="maxVertex">-</div></div></div></header>
<div class="controls"><label>Data <input id="dataFile" type="file" accept="application/json,.json"/></label><button id="reloadData" type="button">Reload JSON</button><span id="status" class="status"></span><label>Unit <select id="unit"><option value="THz">THz</option><option value="meV">meV</option></select></label><label>Color <select id="colorBy"><option value="weight">Magnon weight</option><option value="chirality">Chirality</option></select></label><label><input id="overlayBare" type="checkbox" checked/> Overlay bare</label><label><input id="showNegative" type="checkbox" checked/> Negative region</label><label>Y max <input id="yMax" type="number" min="0" step="0.1" placeholder="auto"/></label><button id="resetZoom" type="button">Reset view</button></div>
<div class="plot-wrap" id="plotWrap"><canvas id="plot"></canvas><div class="tooltip" id="tooltip"></div></div><div class="footer"><div class="legend"><span class="swatch"><span class="line"></span> hybrid</span><span class="swatch"><span class="line mag"></span> bare magnon</span><span class="swatch"><span class="line ph"></span> bare phonon</span></div><div id="sourceLabel"></div></div></section></main>
<script>
let DATA=null;const DATA_URL={data_url_json};const canvas=document.getElementById('plot'),wrap=document.getElementById('plotWrap'),tip=document.getElementById('tooltip');
const unitEl=document.getElementById('unit'),colorEl=document.getElementById('colorBy'),overlayEl=document.getElementById('overlayBare'),negEl=document.getElementById('showNegative'),yMaxEl=document.getElementById('yMax'),resetZoomEl=document.getElementById('resetZoom'),statusEl=document.getElementById('status'),fileEl=document.getElementById('dataFile'),reloadEl=document.getElementById('reloadData');
const state={{mouse:null,plot:null,view:null,dragging:false,dragStart:null}};
function fmt(x,n=3){{return Number(x).toFixed(n).replace(/\\.?0+$/,'')}}function toUnit(v){{return unitEl.value==='THz'?v/DATA.mevPerThz:v}}function unitLabel(){{return unitEl.value==='THz'?'Frequency (THz)':'Energy (meV)'}}function energyArray(a){{return a.map(r=>r.map(toUnit))}}function clamp(v,a,b){{return Math.max(a,Math.min(b,v))}}function lerp(a,b,t){{return a+(b-a)*t}}
function colorWeight(t){{t=clamp(t,0,1);const s=[[68,1,84],[59,82,139],[33,145,140],[94,201,98],[253,231,37]],x=t*(s.length-1),i=Math.min(s.length-2,Math.floor(x)),f=x-i,c=s[i].map((v,j)=>Math.round(lerp(v,s[i+1][j],f)));return`rgb(${{c[0]}},${{c[1]}},${{c[2]}})`}}function colorChirality(t){{t=clamp((t+1)*.5,0,1);const a=t<.5?[49,54,149]:[255,255,255],b=t<.5?[255,255,255]:[165,0,38],f=t<.5?t*2:(t-.5)*2,c=a.map((v,j)=>Math.round(lerp(v,b[j],f)));return`rgb(${{c[0]}},${{c[1]}},${{c[2]}})`}}function seriesColor(v){{return colorEl.value==='chirality'?colorChirality(v):colorWeight(v)}}
function setupText(){{document.getElementById('pathLabel').textContent=DATA.tickLabels.join(' → ');document.getElementById('component').textContent=DATA.component.toUpperCase();document.getElementById('modeCount').textContent=`${{DATA.stats.nHybrid}} = ${{DATA.stats.nMagnon}} mag + ${{DATA.stats.nPhonon}} ph`;document.getElementById('energyRange').textContent=`${{fmt(DATA.stats.emin)}}–${{fmt(DATA.stats.emax)}} meV`;document.getElementById('maxVertex').textContent=`${{fmt(DATA.stats.maxVertexMev)}} meV`;document.getElementById('sourceLabel').textContent=DATA.source||DATA_URL}}
async function loadDataFromUrl(){{statusEl.textContent='loading JSON';const r=await fetch(DATA_URL,{{cache:'no-store'}});if(!r.ok)throw new Error(`${{r.status}} ${{r.statusText}}`);DATA=await r.json();statusEl.textContent='';state.view=null;setupText();draw()}}function loadDataObject(o,n){{DATA=o;if(n)DATA.source=n;statusEl.textContent='';state.view=null;setupText();draw()}}
function ranges(h,bm,bp){{let y=h.flat();if(overlayEl.checked)y=y.concat(bm.flat(),bp.flat());let ymin=negEl.checked?Math.min(0,Math.min(...y)):0,ymax=Math.max(...y);if(yMaxEl.value!=='')ymax=Number(yMaxEl.value);const p=Math.max((ymax-ymin)*.05,.02);return{{x0:DATA.kdist[0],x1:DATA.kdist[DATA.kdist.length-1],ymin:ymin-(negEl.checked?p:0),ymax:ymax+p}}}}function resetZoom(){{state.view=null;draw()}}
function draw(){{if(!DATA)return;const rect=wrap.getBoundingClientRect(),dpr=window.devicePixelRatio||1;canvas.width=Math.max(600,Math.floor(rect.width*dpr));canvas.height=Math.max(360,Math.floor(rect.height*dpr));const ctx=canvas.getContext('2d');ctx.setTransform(dpr,0,0,dpr,0,0);const W=canvas.width/dpr,H=canvas.height/dpr;ctx.clearRect(0,0,W,H);const left=76,right=86,top=22,bottom=66,pw=W-left-right,ph=H-top-bottom;const hy=energyArray(DATA.hybrid),bm=energyArray(DATA.bareMagnon),bp=energyArray(DATA.barePhonon),rr=ranges(hy,bm,bp);if(!state.view)state.view={{...rr}};const{{x0,x1,ymin,ymax}}=state.view;const X=x=>left+(x-x0)/(x1-x0)*pw,Y=y=>top+(ymax-y)/(ymax-ymin)*ph;state.plot={{left,top,pw,ph,x0,x1,ymin,ymax,X,Y}};ctx.fillStyle='#fff';ctx.fillRect(0,0,W,H);ctx.strokeStyle='#d8dde6';ctx.lineWidth=1;ctx.fillStyle='#5b6472';ctx.font='12px Inter, sans-serif';ctx.textAlign='right';ctx.textBaseline='middle';for(let i=0;i<=5;i++){{const y=ymin+(ymax-ymin)*i/5,py=Y(y);ctx.beginPath();ctx.moveTo(left,py);ctx.lineTo(left+pw,py);ctx.stroke();ctx.fillText(fmt(y,2),left-10,py)}}for(let i=0;i<DATA.tickIndices.length;i++){{const idx=DATA.tickIndices[i],px=X(DATA.kdist[idx]);if(px<left-20||px>left+pw+20)continue;ctx.strokeStyle='#cfd6e2';ctx.beginPath();ctx.moveTo(px,top);ctx.lineTo(px,top+ph);ctx.stroke();ctx.fillStyle='#15171a';ctx.font='16px Inter, sans-serif';ctx.textAlign='center';ctx.textBaseline='top';ctx.fillText(DATA.tickLabels[i],px,top+ph+18)}}ctx.strokeStyle='#222831';ctx.lineWidth=1.2;ctx.beginPath();ctx.moveTo(left,top);ctx.lineTo(left,top+ph);ctx.lineTo(left+pw,top+ph);ctx.stroke();ctx.save();ctx.translate(20,top+ph/2);ctx.rotate(-Math.PI/2);ctx.fillStyle='#15171a';ctx.font='15px Inter, sans-serif';ctx.textAlign='center';ctx.fillText(unitLabel(),0,0);ctx.restore();ctx.save();ctx.beginPath();ctx.rect(left,top,pw,ph);ctx.clip();function plain(a,color,dash,lw,alpha){{ctx.save();ctx.globalAlpha=alpha;ctx.strokeStyle=color;ctx.lineWidth=lw;ctx.setLineDash(dash||[]);for(let b=0;b<a[0].length;b++){{ctx.beginPath();for(let i=0;i<a.length;i++){{const px=X(DATA.kdist[i]),py=Y(a[i][b]);if(i===0)ctx.moveTo(px,py);else ctx.lineTo(px,py)}}ctx.stroke()}}ctx.restore()}}if(overlayEl.checked){{plain(bp,'#8a94a6',[],.8,.55);plain(bm,'#b91c1c',[6,4],1.3,.72)}}const cv=colorEl.value==='chirality'?DATA.hybridChirality:DATA.magnonWeight;for(let b=0;b<hy[0].length;b++)for(let i=0;i<hy.length-1;i++){{const v=.5*(cv[i][b]+cv[i+1][b]);ctx.strokeStyle=seriesColor(v);ctx.lineWidth=1.45;ctx.beginPath();ctx.moveTo(X(DATA.kdist[i]),Y(hy[i][b]));ctx.lineTo(X(DATA.kdist[i+1]),Y(hy[i+1][b]));ctx.stroke()}}ctx.restore();colorbar(ctx,W,top,ph);if(state.mouse)hover(ctx,hy,cv)}}
function colorbar(ctx,W,top,ph){{const x=W-56,y=top+8,h=Math.min(230,ph*.55),w=12;for(let i=0;i<h;i++){{const t=1-i/(h-1),v=colorEl.value==='chirality'?-1+2*t:t;ctx.fillStyle=seriesColor(v);ctx.fillRect(x,y+i,w,1)}}ctx.strokeStyle='#222831';ctx.strokeRect(x,y,w,h);ctx.fillStyle='#15171a';ctx.font='11px Inter, sans-serif';ctx.textAlign='left';ctx.textBaseline='middle';ctx.fillText(colorEl.value==='chirality'?'+1':'1',x+18,y+2);ctx.fillText(colorEl.value==='chirality'?'-1':'0',x+18,y+h-2);ctx.save();ctx.translate(x+42,y+h/2);ctx.rotate(-Math.PI/2);ctx.textAlign='center';ctx.fillText(colorEl.value==='chirality'?'Chirality':'Magnon weight',0,0);ctx.restore()}}
function hover(ctx,hy,cv){{const p=state.plot,m=state.mouse;if(!p||m.x<p.left||m.x>p.left+p.pw||m.y<p.top||m.y>p.top+p.ph){{tip.style.display='none';return}}const tx=p.x0+(m.x-p.left)/p.pw*(p.x1-p.x0);let iq=0,best=Infinity;for(let i=0;i<DATA.kdist.length;i++){{const d=Math.abs(DATA.kdist[i]-tx);if(d<best){{best=d;iq=i}}}}let ib=0,bd=Infinity;for(let b=0;b<hy[iq].length;b++){{const d=Math.abs(p.Y(hy[iq][b])-m.y);if(d<bd){{bd=d;ib=b}}}}const px=p.X(DATA.kdist[iq]),py=p.Y(hy[iq][ib]);ctx.fillStyle='#111827';ctx.beginPath();ctx.arc(px,py,4,0,Math.PI*2);ctx.fill();const q=DATA.qpts[iq];tip.style.display='block';tip.style.left=`${{m.x}}px`;tip.style.top=`${{m.y}}px`;tip.innerHTML=`<strong>mode ${{ib}}</strong><br>q = (${{fmt(q[0],3)}}, ${{fmt(q[1],3)}}, ${{fmt(q[2],3)}})<br>${{unitLabel()}} = ${{fmt(hy[iq][ib],4)}}<br>magnon weight = ${{fmt(DATA.magnonWeight[iq][ib],4)}}<br>chirality = ${{fmt(DATA.hybridChirality[iq][ib],4)}}`}}
function pt(e){{const r=canvas.getBoundingClientRect();return{{x:e.clientX-r.left,y:e.clientY-r.top}}}}function inside(a){{const p=state.plot;return p&&a.x>=p.left&&a.x<=p.left+p.pw&&a.y>=p.top&&a.y<=p.top+p.ph}}function dat(a){{const p=state.plot;return{{x:p.x0+(a.x-p.left)/p.pw*(p.x1-p.x0),y:p.ymax-(a.y-p.top)/p.ph*(p.ymax-p.ymin)}}}}function zoom(a,s){{if(!inside(a))return;const p=state.plot,d=dat(a),xf=(d.x-p.x0)/(p.x1-p.x0),yf=(d.y-p.ymin)/(p.ymax-p.ymin),xs=(p.x1-p.x0)*s,ys=(p.ymax-p.ymin)*s;state.view={{x0:d.x-xs*xf,x1:d.x+xs*(1-xf),ymin:d.y-ys*yf,ymax:d.y+ys*(1-yf)}};draw()}}function pan(dx,dy){{const p=state.plot;if(!p)return;const xs=-dx/p.pw*(p.x1-p.x0),ys=dy/p.ph*(p.ymax-p.ymin);state.view={{x0:p.x0+xs,x1:p.x1+xs,ymin:p.ymin+ys,ymax:p.ymax+ys}};draw()}}
canvas.addEventListener('wheel',e=>{{e.preventDefault();zoom(pt(e),Math.exp(e.deltaY*.0012))}},{{passive:false}});canvas.addEventListener('mousedown',e=>{{const p=pt(e);if(!inside(p))return;state.dragging=true;state.dragStart=p;canvas.style.cursor='grabbing'}});window.addEventListener('mouseup',()=>{{state.dragging=false;state.dragStart=null;canvas.style.cursor='default'}});canvas.addEventListener('mousemove',e=>{{const p=pt(e);if(state.dragging&&state.dragStart){{pan(p.x-state.dragStart.x,p.y-state.dragStart.y);state.dragStart=p;return}}state.mouse=p;canvas.style.cursor=inside(p)?'crosshair':'default';draw()}});canvas.addEventListener('mouseleave',()=>{{state.mouse=null;tip.style.display='none';if(!state.dragging)canvas.style.cursor='default';draw()}});
resetZoomEl.addEventListener('click',resetZoom);unitEl.addEventListener('input',resetZoom);overlayEl.addEventListener('input',resetZoom);negEl.addEventListener('input',resetZoom);yMaxEl.addEventListener('input',resetZoom);colorEl.addEventListener('input',draw);window.addEventListener('resize',draw);reloadEl.addEventListener('click',()=>loadDataFromUrl().catch(()=>{{statusEl.textContent='auto-load failed; choose JSON file'}}));fileEl.addEventListener('change',async e=>{{const f=e.target.files&&e.target.files[0];if(!f)return;loadDataObject(JSON.parse(await f.text()),f.name)}});loadDataFromUrl().catch(()=>{{statusEl.textContent='choose JSON file'}});
</script></body></html>"""


def save_hybrid_html(path, *, data_url: str = "hybrid_plot_data.json") -> None:
    Path(path).write_text(_html_template(data_url), encoding="utf-8")


def save_hybrid_interactive_html(
    html_path,
    data,
    *,
    data_path=None,
    kdist=None,
    tick_indices=None,
    kpath_symbols=None,
    source=None,
) -> tuple[str, str]:
    html_path = Path(html_path)
    if data_path is None:
        data_path = html_path.with_name(f"{html_path.stem}_data.json")
    data_path = Path(data_path)
    os.makedirs(html_path.parent or Path("."), exist_ok=True)
    os.makedirs(data_path.parent or Path("."), exist_ok=True)
    save_hybrid_html_data(
        data_path,
        data,
        kdist=kdist,
        tick_indices=tick_indices,
        kpath_symbols=kpath_symbols,
        source=source,
    )
    try:
        rel_data = os.path.relpath(data_path, start=html_path.parent)
    except ValueError:
        rel_data = str(data_path)
    save_hybrid_html(html_path, data_url=rel_data)
    return str(html_path), str(data_path)
