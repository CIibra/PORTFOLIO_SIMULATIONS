-- Script : analyse_retards_transport.sql
-- Auteur : CISSE Ibrahim
-- Objectif : Analyser les retards dans un réseau de transport (train, bus, métro) pour identifier les lignes et horaires les plus impactés.

-- Étapes réalisées :
-- 1. Calcul du retard moyen par ligne
-- 2. Identification des lignes les plus en retard
-- 3. Analyse des retards par tranche horaire
-- 4. Sélection des trajets avec retard supérieur à 15 minutes

-- 1. Retard moyen par ligne
SELECT 
  ligne_id,
  ROUND(AVG(retard_minutes), 2) AS retard_moyen
FROM trajets
WHERE retard_minutes IS NOT NULL
GROUP BY ligne_id
ORDER BY retard_moyen DESC;

-- 2. Lignes les plus en retard (top 5)
SELECT 
  ligne_id,
  COUNT(*) AS nb_trajets,
  ROUND(AVG(retard_minutes), 2) AS retard_moyen
FROM trajets
WHERE retard_minutes IS NOT NULL
GROUP BY ligne_id
ORDER BY retard_moyen DESC
LIMIT 5;

-- 3. Retards par tranche horaire
SELECT 
  CASE
    WHEN EXTRACT(HOUR FROM heure_depart) BETWEEN 6 AND 9 THEN 'Matinée'
    WHEN EXTRACT(HOUR FROM heure_depart) BETWEEN 16 AND 19 THEN 'Soirée'
    ELSE 'Autres'
  END AS tranche_horaire,
  ROUND(AVG(retard_minutes), 2) AS retard_moyen
FROM trajets
WHERE retard_minutes IS NOT NULL
GROUP BY tranche_horaire;

-- 4. Trajets avec retard > 15 minutes
SELECT 
  trajet_id,
  ligne_id,
  heure_depart,
  retard_minutes
FROM trajets
WHERE retard_minutes > 15
ORDER BY retard_minutes DESC;
