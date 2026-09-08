create index if not exists idx_races_external_id on public.races(external_id);
create index if not exists idx_participants_race_id on public.participants(race_id);
create index if not exists idx_participants_finish_position on public.participants(finish_position);
create unique index if not exists uq_participants_race_horse_number
  on public.participants(race_id, horse_number)
  where horse_number is not null;

create or replace function public.get_training_history_until(p_race_date date, p_limit int default 200000)
returns table (
  race_id text, race_date date, discipline text, hippodrome text, distance int,
  terrain text, field_size int, horse_number int, horse_name text, jockey text,
  trainer text, odds numeric, draw int, weight numeric, recent_form text,
  finish_position int
)
language sql stable as $$
  select r.external_id, r.race_date, r.discipline, r.hippodrome, r.distance_m,
         r.terrain, r.field_size, p.horse_number, p.horse_name,
         p.jockey_name, p.trainer_name, p.odds, p.draw, p.weight_kg,
         p.recent_form, p.finish_position
  from public.participants p
  join public.races r on r.id = p.race_id
  where r.race_date < p_race_date
    and p.finish_position is not null
    and coalesce(p.is_non_runner,false) = false
  order by r.race_date asc, r.external_id asc, p.horse_number asc
  limit p_limit;
$$;
