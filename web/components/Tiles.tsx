"use client";
import s from "@/app/page.module.css";
import type { TileSpec } from "@/lib/tiles";

export function Tiles({ tiles, big }: { tiles: TileSpec[]; big?: boolean }) {
  return (
    <div className={`${s.tiles} ${big ? s.tilesBig : ""}`}>
      {tiles.map((x, i) => (
        <div key={i} className={s.tile}>
          <div className={`lbl ${s.tileL}`}>{x.label}</div>
          <div className={`num ${s.tileV}`}>{x.value}</div>
          {x.delta && <div className={`num ${s.tileD} ${x.good === true ? s.good : x.good === false ? s.bad : ""}`}>{x.delta}</div>}
        </div>
      ))}
    </div>
  );
}
