create table if not exists public.courses (
  id bigint generated always as identity primary key,
  date date not null,
  heure text,
  hippodrome text,
  discipline text,
  distance text,
  statut text,
  partants integer,
  prix numeric,
  reunion_num text not null,
  course_num text not null,
  corde text,
  categorie text,
  conditions text,
  meteo_json jsonb,
  type_pari text,
  ordre_arrivee text,
  rapports_json jsonb,
  created_at timestamptz not null default now(),
  unique(date, reunion_num, course_num)
);

create table if not exists public.participants (
  id bigint generated always as identity primary key,
  course_id bigint not null references public.courses(id) on delete cascade,
  nom text,
  numero integer,
  sexe text,
  age integer,
  musique text,
  gains numeric,
  driver text,
  entraineur text,
  proprietaire text,
  ferrage text,
  oeilleres text,
  nb_courses integer,
  nb_victoires integer,
  nb_places integer,
  cote_ref numeric,
  statut text,
  prediction_score numeric,
  classement integer
);

create index if not exists idx_courses_date on public.courses(date);
create index if not exists idx_participants_course on public.participants(course_id);
create index if not exists idx_participants_prediction on public.participants(prediction_score desc);

alter table public.courses enable row level security;
alter table public.participants enable row level security;

-- MVP: lecture/écriture via la clé serveur stockée dans Streamlit Secrets.
-- Si l'application devient publique, ajouter une authentification et des politiques RLS plus fines.
