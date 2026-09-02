# IPCC WGI AR6 reference regions — data licence and citation

`IPCC-WGI-reference-regions-v4.geojson` (695,765 bytes) is redistributed
unmodified from the IPCC WGI Atlas repository:

<https://github.com/IPCC-WG1/Atlas/blob/main/reference-regions/IPCC-WGI-reference-regions-v4.geojson>

**Licence:** Creative Commons Attribution 4.0 International (CC BY 4.0),
<https://creativecommons.org/licenses/by/4.0/>.

**Required citation:**

> Iturbide, M., Fernández, J., Gutiérrez, J.M. et al. Implementation of FAIR
> principles in the IPCC: the WGI AR6 Atlas repository. *Scientific Data* 9, 629
> (2022). <https://doi.org/10.1038/s41597-022-01739-y>

The 58 features carry `Acronym`, `Name`, `Type` (`Land` / `Land-Ocean` /
`Ocean`) and `Continent`. `tools/ar6_regions.py` reads the 46 land-touching
ones (`Land` + `Land-Ocean`), the same set regionmask exposed as
`ar6.land`; verified point-for-point against regionmask on a 2-degree global
grid, 0 disagreements.

The rest of this repository is MIT-licensed (see `../../LICENSE`); this data
file is not.
